rm(list=ls())

#Campaign retailer1 ----
install.packages("arrow")
library(arrow)
library(dplyr)
library(lubridate)

#specify the path and load files
file_path <- "~/Desktop/parquet files/part-00000-ffbea56e-a67e-49e8-b7c5-155ffa2d97de-c000.snappy.parquet"
part_0_rt1 <- arrow::read_parquet(file_path)

file_path1 <- "~/Desktop/parquet files/part-00001-ffbea56e-a67e-49e8-b7c5-155ffa2d97de-c000.snappy.parquet"
part_1_rt1 <- arrow::read_parquet(file_path1)

file_path2 <- "~/Desktop/parquet files/part-00002-ffbea56e-a67e-49e8-b7c5-155ffa2d97de-c000.snappy.parquet"
part_2_rt1 <- arrow::read_parquet(file_path2)

file_path3 <- "~/Desktop/parquet files/part-00003-ffbea56e-a67e-49e8-b7c5-155ffa2d97de-c000.snappy.parquet"
part_3_rt1 <- arrow::read_parquet(file_path3)

file_path4 <- "~/Desktop/parquet files/part-00004-ffbea56e-a67e-49e8-b7c5-155ffa2d97de-c000.snappy.parquet"
part_4_rt1 <- arrow::read_parquet(file_path4)

####Data cleaning and engineering retailer1
#remove data in excess
columns_to_remove <- c(1, 6, 7)
part_0_rt1 <- part_0_rt1[, -columns_to_remove]

part_1_rt1 <- part_1_rt1[, -columns_to_remove]

part_2_rt1 <- part_2_rt1[, -columns_to_remove]

part_3_rt1 <- part_3_rt1[, -columns_to_remove]

part_4_rt1 <- part_4_rt1[, -columns_to_remove]

#combine all parts into a unique dataset for retailer1
stacked_dataset1 <- rbind(part_0_rt1, part_1_rt1, part_2_rt1, part_3_rt1, part_4_rt1)
#adjust date
stacked_dataset1$date_code <- ymd(stacked_dataset1$date_code, truncated = 8)

#exploration, descriptives, plots, etc...
summary(stacked_dataset1)
mean(stacked_dataset1$quantity_sold)
mean(stacked_dataset1$revenue_after_discount_incl_vat)
median(stacked_dataset1$revenue_after_discount_incl_vat)

unique_store_types <- unique(stacked_dataset1$store_type_generic)
print(unique_store_types)

#create a new column 'online_store' with 1 for 'online' and 0 for others
stacked_dataset1 <- stacked_dataset1 %>%
  mutate(online_store = ifelse(store_type_generic == "Online", 1, 0))

#create a new dataset excluding observations with 'period' equal to 4
retailer1_beforeLP <- subset(stacked_dataset1, period_code != 4)

#sanity check: number of unique customers before aggregation
length(unique(retailer1_beforeLP$customer_code)) #42480

#aggregating data on customer level and creation of new variables
retailer1_beforeLP <- retailer1_beforeLP %>%
  group_by(customer_code) %>%
  summarise(
    total_revenue = sum(revenue_after_discount_incl_vat),
    total_quantity = sum(quantity_sold),
    trips = n(),
    last_purchase_date = max(date_code),
    average_revenue_per_purchase = total_revenue / trips,
    Online_purchase_ratio = pmin(round(sum(online_store) / trips, 3), 1),
    average_quantity_per_purchase = sum(total_quantity) / trips,
    last_expenditure = sum(ifelse(period_code == 3, revenue_after_discount_incl_vat, 0)),
    redeemer_latest_ind = unique(redeemer_latest_ind)
  )
#sanity check: number of observation should equal the number of unique customers before the aggregation: 42480 obs means the aggregation went correctly

#rearrangement of driving variables in deciles
#expenditure_level
retailer1_beforeLP <- retailer1_beforeLP %>%
  mutate(expenditure_level = ntile(average_revenue_per_purchase, 10))

#quantity_purchased_level
retailer1_beforeLP <- retailer1_beforeLP %>%
  mutate(quantity_purchased_level = ntile(average_quantity_per_purchase, 10))

#last_expenditure_level
retailer1_beforeLP <- retailer1_beforeLP %>%
  mutate(last_expenditure_level = ntile(last_expenditure, 10))

#trips_level
retailer1_beforeLP <- retailer1_beforeLP %>%
  mutate(trips_level = ntile(trips, 10))

#online_purch_level
retailer1_beforeLP <- retailer1_beforeLP %>%
  mutate(online_purch_level = ntile(Online_purchase_ratio, 10))

####Bagging trees retailer1
library(ipred)
install.packages("caret")
library(caret)
install.packages("rpart")
library(rpart)

#estimation sample (85%) and test sample (15%)
set.seed(123)
total_observations <- nrow(retailer1_beforeLP)
estimation_sample_size <- round(0.85 * total_observations)
estimation_sample_indices <- sample(1:total_observations, size = estimation_sample_size, replace = FALSE)
estimation_sample <- retailer1_beforeLP[estimation_sample_indices, ]
summary(estimation_sample)
test_sample <- retailer1_beforeLP[-estimation_sample_indices, ]

#Training
#define a range of values for the number of trees (nbagg)
num_trees <- c(50, 100, 150)
#Bagging_tree1 (online_purch_level included)
Bagging_tree1 <- train(as.factor(redeemer_latest_ind) ~ trips_level + expenditure_level + quantity_purchased_level + last_expenditure_level + online_purch_level, data=estimation_sample,
                       method="treebag", trControl = trainControl(method = "cv", number = 5), n_tree = num_trees)
#Bagging_tree2 (online_purch_level excluded)
Bagging_tree2 <- train(as.factor(redeemer_latest_ind) ~ trips_level + expenditure_level + quantity_purchased_level + last_expenditure_level, data=estimation_sample,
                       method="treebag", trControl = trainControl(method = "cv", number = 5), n_tree= num_trees)
#inspect trees
Bagging_tree1 #accuracy 0.884
Bagging_tree2 #accuracy 0.898 interesting! (more accurate without online_purch_level)

#Predict
predictions_bagging1 <- predict(Bagging_tree1, newdata=test_sample,
                                type ="prob")
predictions_bagging2 <- predict(Bagging_tree2, newdata=test_sample,
                                type ="prob")
#sanity check: number of columns of predictions should be as the number of observations in the test_sample
nrow(predictions_bagging1) #6372
nrow(predictions_bagging2) #6372

#calculate variable importance
pred.imp1 <- varImp(Bagging_tree1)
pred.imp1
pred.imp2 <- varImp(Bagging_tree2)
pred.imp2 
#plot variable importance
barplot(pred.imp1$importance$Overall, names.arg = row.names(pred.imp1$importance))
barplot(pred.imp2$importance$Overall, names.arg = row.names(pred.imp2$importance))

#Predictive power of the models (hit rate, top decile lift, Gini coefficient)
#Predictions_bagging1 and predictions_bagging2 against test_sample
#make the basis for the hit rate table
predictions_bagging1 <- predictions_bagging1[, -1]
predictions_bagging2 <- predictions_bagging2[, -1]
predicted_model1 <- ifelse(predictions_bagging1>.5,1,0)
predicted_model2 <- ifelse(predictions_bagging2>.5,1,0)

hit_rate_model1 <- table(test_sample$redeemer_latest_ind, predicted_model1, dnn= c("Observed", "Predicted"))
hit_rate_model2 <- table(test_sample$redeemer_latest_ind, predicted_model2, dnn= c("Observed", "Predicted"))

hit_rate_model1
hit_rate_model2
#get the hit rate
(hit_rate_model1[1,1]+hit_rate_model1[2,2])/sum(hit_rate_model1) #very good predictive power: 0.889
(hit_rate_model2[1,1]+hit_rate_model2[2,2])/sum(hit_rate_model2) #very good predictive power: 0.901

#TDL table
decile_predicted_model1 <- ntile(predicted_model1, 10)
decile_predicted_model2 <- ntile(predicted_model2, 10)
decile_model1 <- table(test_sample$redeemer_latest_ind, decile_predicted_model1, dnn=
                           c("Observed", "Decile"))
decile_model2 <- table(test_sample$redeemer_latest_ind, decile_predicted_model2, dnn=
                         c("Observed", "Decile"))
decile_model1
decile_model2
#get the TDL
(decile_model1[2,10] / (decile_model1[1,10]+ decile_model1[2,10])) / mean(test_sample$redeemer_latest_ind) #very good: predicts 4.648 times better than random selection
(decile_model2[2,10] / (decile_model2[1,10]+ decile_model2[2,10])) / mean(test_sample$redeemer_latest_ind) #very good: predicts 5.031 times better than random selection

#Gini coefficient
library(ROCR)
install.packages("ROCR")
pred_model1 <- prediction(predicted_model1, test_sample$redeemer_latest_ind)
perf_model1 <- performance(pred_model1,"tpr","fpr")
plot(perf_model1,xlab="Cumulative % of observations",ylab="Cumulative % of positive cases",xlim=c(0,1),ylim=c(0,1),xaxs="i",yaxs="i")
abline(0,1)
auc_model1 <- performance(pred_model1,"auc")
as.numeric(auc_model1@y.values)*2-1 #0.484: closer to 1 the better (if it is 0 that means no better predictions than random selection)

pred_model2 <- prediction(predicted_model2, test_sample$redeemer_latest_ind)
perf_model2 <- performance(pred_model2,"tpr","fpr")
plot(perf_model2,xlab="Cumulative % of observations",ylab="Cumulative % of positive cases",xlim=c(0,1),ylim=c(0,1),xaxs="i",yaxs="i")
abline(0,1)
auc_model2 <- performance(pred_model2,"auc")
as.numeric(auc_model2@y.values)*2-1 #0.481: closer to 1 the better (if it is 0 that means no better predictions than random selection)


#Campaigns retailer2 (prog.1 & prog.2) ----
#specify path and load files (retailer 2 program 1, retailer 2 program 2)
file_path5 <- "~/Desktop/parquet files/part-00000-2f5927ef-cceb-40ba-b580-e6496922ddf4-c000.snappy.parquet"
retailer2_prog1 <- arrow::read_parquet(file_path5)

file_path6 <- "~/Desktop/parquet files/part-00000-a05211a0-176d-4c71-bfd0-7c101720ca7d-c000.snappy.parquet"
retailer2_prog2 <- arrow::read_parquet(file_path6)

####Data cleaning and engineering retailer2_prog1
#remove data in excess
retailer2_prog1 <- subset(retailer2_prog1, period_code != 4)
#aggregation on customer level and creation of driving variables
retailer2_prog1_before <- retailer2_prog1 %>%
  group_by(customer_card_number) %>%
  summarise(
    trips = n(),
    total_quantity = sum(quantity_sold),
    tot_revenue = sum(revenue_after_discount_incl_vat),
    quantity_purchased_level = total_quantity / trips,
    expenditure_level = tot_revenue / trips,
    last_expenditure = sum(ifelse(period_code == 3, revenue_after_discount_incl_vat, 0)),
    Redeemer_ind = unique(Redeemer_ind)
  )
#rearrangement of driving variables in deciles
#expenditure_level
retailer2_prog1_before <- retailer2_prog1_before %>%
  mutate(expenditure_level = ntile(expenditure_level, 10))

#quantity_purchased_level
retailer2_prog1_before <- retailer2_prog1_before %>%
  mutate(quantity_purchased_level = ntile(quantity_purchased_level, 10))

#last_expenditure_level
retailer2_prog1_before <- retailer2_prog1_before %>%
  mutate(last_expenditure_level = ntile(last_expenditure, 10))

#trips_level
retailer2_prog1_before <- retailer2_prog1_before %>%
  mutate(trips_level = ntile(trips, 10))

#renaming vars to apply bagging_tree2
colnames(retailer2_prog1_before)[colnames(retailer2_prog1_before) == "Redeemer_ind"] <- "redeemer_latest_ind"

####Transfer learning: use Bagging_tree_2 on retailer2_prog1
#predictions
predictions_bagging_retailer1_p1 <- predict(Bagging_tree2, newdata=retailer2_prog1_before,
                                type ="prob")
predictions_bagging_retailer1_p1 <- predictions_bagging_retailer1_p1[, -1]

predicted_model3 <- ifelse(predictions_bagging_retailer1_p1>.5,1,0)

#How did the predictions go? Bagging_tree2 against obeserved redemptions in retailer2_prog1
#make the basis for the hit rate table
hit_rate_model3 <- table(retailer2_prog1_before$redeemer_latest_ind, predicted_model3, dnn= c("Observed", "Predicted"))

hit_rate_model3
#get the hit rate
(hit_rate_model3[1,1]+hit_rate_model3[2,2])/sum(hit_rate_model3) #0.874

#top-decile lift
#TDL table
decile_predicted_model3 <- ntile(predicted_model3, 10)
decile_model3 <- table(retailer2_prog1_before$redeemer_latest_ind, decile_predicted_model3, dnn=
                         c("Observed", "Decile"))
decile_model3

#get the TDL
(decile_model3[2,10] / (decile_model3[1,10]+ decile_model3[2,10])) / mean(retailer2_prog1_before$redeemer_latest_ind) #3.94: lowered by more than 1 point, but still a good performance

#Gini coefficient
pred_model3 <- prediction(predicted_model3, retailer2_prog1_before$redeemer_latest_ind)
perf_model3 <- performance(pred_model3,"tpr","fpr")
plot(perf_model3,xlab="Cumulative % of observations",ylab="Cumulative % of positive cases",xlim=c(0,1),ylim=c(0,1),xaxs="i",yaxs="i")
abline(0,1)
auc_model3 <- performance(pred_model3,"auc")
as.numeric(auc_model3@y.values)*2-1 #0.346: it's lower than Gini model 2, but still higher than 0 meaning better predictions than random selection


####Bagging tree retailer 2 prog 1
#create estimation_sample2 (85%) and test_sample2 (15%)
set.seed(123)
total_observations_new <- nrow(retailer2_prog1_before)
estimation_sample_size_new <- round(0.85 * total_observations_new)
estimation_sample_indices_new <- sample(1:total_observations_new, size = estimation_sample_size_new, replace = FALSE)
estimation_sample2 <- retailer2_prog1_before[estimation_sample_indices_new, ]
summary(estimation_sample2)
test_sample2 <- retailer2_prog1_before[-estimation_sample_indices_new, ]

#Bagging_tree3
Bagging_tree3 <- train(as.factor(redeemer_latest_ind) ~ trips_level + expenditure_level + quantity_purchased_level + last_expenditure_level, data=estimation_sample2,
                       method="treebag", trControl = trainControl(method = "cv", number = 5), n_tree = num_trees)

#inspect tree
Bagging_tree3 #0.88

#predict on test data from same dataset
predictions_bagging3 <- predict(Bagging_tree3, newdata=test_sample2,
                                type ="prob")

#sanity check: number of columns of predictions should be as the number of observations in the test_sample2
nrow(predictions_bagging3) #18511

#calculate variable importance
pred.imp3 <- varImp(Bagging_tree3)
pred.imp3
#plot the results
barplot(pred.imp3$importance$Overall, names.arg = row.names(pred.imp3$importance))

#Predictive power of the model (hit rate, top decile lift, Gini coefficient)
#Predictions_bagging3 against test_sample2
#make the basis for the hit rate table
predictions_bagging3 <- predictions_bagging3[, -1]
predicted_model4 <- ifelse(predictions_bagging3>.5,1,0)

hit_rate_model4 <- table(test_sample2$redeemer_latest_ind, predicted_model4, dnn= c("Observed", "Predicted"))

hit_rate_model4
#get the hit rate
(hit_rate_model4[1,1]+hit_rate_model4[2,2])/sum(hit_rate_model4) #0.88

#top-decile lift
#TDL table
decile_predicted_model4 <- ntile(predicted_model4, 10)
decile_model4 <- table(test_sample2$redeemer_latest_ind, decile_predicted_model4, dnn=
                         c("Observed", "Decile"))
decile_model4
#get the TDL
(decile_model4[2,10] / (decile_model4[1,10]+ decile_model4[2,10])) / mean(test_sample2$redeemer_latest_ind) #3.807 times better than random selection at predicting top cases

#Gini coefficient
pred_model4 <- prediction(predicted_model4, test_sample2$redeemer_latest_ind)
perf_model4 <- performance(pred_model4,"tpr","fpr")
plot(perf_model4,xlab="Cumulative % of observations",ylab="Cumulative % of positive cases",xlim=c(0,1),ylim=c(0,1),xaxs="i",yaxs="i")
abline(0,1)
auc_model4 <- performance(pred_model4,"auc")
as.numeric(auc_model4@y.values)*2-1 #0.34: closer to 1 the better (still above 0 meaning better than random selection)

####Cleaning and rearranging retailer2_prog2
#remove data
retailer2_prog2 <- subset(retailer2_prog2, period_code != 4)
#aggregate on customer level and create driving variables
retailer2_prog2_before <- retailer2_prog2 %>%
  group_by(customer_card_number) %>%
  summarise(
    trips = n(),
    total_quantity = sum(quantity_sold),
    tot_revenue = sum(revenue_after_discount_incl_vat),
    quantity_level = total_quantity / trips,
    expenditure_level = tot_revenue / trips,
    last_expenditure = sum(ifelse(period_code == 3, revenue_after_discount_incl_vat, 0)),
    redeemer_latest_ind = unique(redeemer_latest_ind)
  )

#rearrange driving variables into deciles
#expenditure_level
retailer2_prog2_before <- retailer2_prog2_before %>%
  mutate(expenditure_level = ntile(expenditure_level, 10))

#quantity_purchased_level
retailer2_prog2_before <- retailer2_prog2_before %>%
  mutate(quantity_purchased_level = ntile(quantity_level, 10))

#last_expenditure_level
retailer2_prog2_before <- retailer2_prog2_before %>%
  mutate(last_expenditure_level = ntile(last_expenditure, 10))

#trips_level
retailer2_prog2_before <- retailer2_prog2_before %>%
  mutate(trips_level = ntile(trips, 10))

#add column in retailer2_prog2_before that tells whether the same customer redeemed in retailer2_prog1 
#first, rename redeemer_latest_ind from program 1 differently
colnames(retailer2_prog1_before)[colnames(retailer2_prog1_before) == "redeemer_latest_ind"] <- "redeemer_program1"


#now, merge datasets
retailer2_prog2_before_joined <- left_join(retailer2_prog2_before,
                                           select(retailer2_prog1_before, customer_card_number, redeemer_program1),
                                           by = "customer_card_number")

#sanity check: retailer2_prog2_before_joined now should have the same number of observation as retailer2_prog2_before
nrow(retailer2_prog2_before_joined) #166141

#rename Redeemer_ind with pastLP_redemption
colnames(retailer2_prog2_before_joined)[colnames(retailer2_prog2_before_joined) == "redeemer_program1"] <- "pastLP_redemption"

#Replace NAs with 0 in the 'pastLP_redemption' column
retailer2_prog2_before_joined$pastLP_redemption[is.na(retailer2_prog2_before_joined$pastLP_redemption)] <- 0

####Transfer learning: use Bagging_tree3 (trained on retailer2 prog1) to predict redemption on retailer2 prog2
predictions_bagging4 <- predict(Bagging_tree3, newdata=retailer2_prog2_before_joined,
                                type ="prob")

#sanity check: number of rows of predictions should be as the number of observations in the retailer2_prog2_before_joined
nrow(predictions_bagging4) #166141

#How did the predictions go? Bagging_tree3 against obeserved redemptions in retailer2_prog2
#make the basis for the hit rate table
predictions_bagging4 <- predictions_bagging4[, -1]
predicted_model5 <- ifelse(predictions_bagging4>.5,1,0)

hit_rate_model5 <- table(retailer2_prog2_before_joined$redeemer_latest_ind, predicted_model5, dnn= c("Observed", "Predicted"))

hit_rate_model5
#get the hit rate
(hit_rate_model5[1,1]+hit_rate_model5[2,2])/sum(hit_rate_model5) #very good predictive power: 0.903

#top-decile lift
#TDL table
decile_predicted_model5 <- ntile(predicted_model5, 10)
decile_model5 <- table(retailer2_prog2_before_joined$redeemer_latest_ind, decile_predicted_model5, dnn=
                         c("Observed", "Decile"))
decile_model5
#get the TDL
(decile_model5[2,10] / (decile_model5[1,10]+ decile_model5[2,10])) / mean(retailer2_prog2_before_joined$redeemer_latest_ind) #4.521 times better than random selection at predicting top cases

#Gini coefficient
pred_model5 <- prediction(predicted_model5, retailer2_prog2_before_joined$redeemer_latest_ind)
perf_model5 <- performance(pred_model5,"tpr","fpr")
plot(perf_model5,xlab="Cumulative % of observations",ylab="Cumulative % of positive cases",xlim=c(0,1),ylim=c(0,1),xaxs="i",yaxs="i")
abline(0,1)
auc_model5 <- performance(pred_model5,"auc")
as.numeric(auc_model5@y.values)*2-1 #0.404: closer to 1 the better (if it is 0 that means no better predictions than random selection)


####Bagging tree retailer 2 prog 2 including pastLP_redemption
#create estimation sample (85%) and test sample (15%)
set.seed(123)
total_observations_new1 <- nrow(retailer2_prog2_before_joined)
estimation_sample_size_new1 <- round(0.85 * total_observations_new1)
estimation_sample_indices_new1 <- sample(1:total_observations_new1, size = estimation_sample_size_new1, replace = FALSE)
estimation_sample3 <- retailer2_prog2_before_joined[estimation_sample_indices_new1, ]
summary(estimation_sample3)
test_sample3 <- retailer2_prog2_before_joined[-estimation_sample_indices_new1, ]

#Bagging_tree4
Bagging_tree4 <- train(as.factor(redeemer_latest_ind) ~ trips_level + expenditure_level + quantity_purchased_level + last_expenditure_level + pastLP_redemption, data=estimation_sample3,
                       method="treebag", trControl = trainControl(method = "cv", number = 5), n_tree = num_trees)

#inspect tree
Bagging_tree4 #0.907

#predict
predictions_bagging5 <- predict(Bagging_tree4, newdata=test_sample3,
                                type ="prob")

#sanity check: number of columns of predictions should equal the number of observations in the test_sample3
nrow(predictions_bagging5) #24921

#calculate variable importance
pred.imp4 <- varImp(Bagging_tree4)
pred.imp4 #interesting: past redemption is very important, as expected
#plot the results
barplot(pred.imp4$importance$Overall, names.arg = row.names(pred.imp4$importance))

#Predictive power of the model (hit rate, top decile lift, Gini coefficient)
#Predictions_bagging4 against test_sample3
#make the basis for the hit rate table
predictions_bagging5 <- predictions_bagging5[, -1]
predicted_model6 <- ifelse(predictions_bagging5>.5,1,0)

hit_rate_model6 <- table(test_sample3$redeemer_latest_ind, predicted_model6, dnn= c("Observed", "Predicted"))

hit_rate_model6
#get the hit rate
(hit_rate_model6[1,1]+hit_rate_model6[2,2])/sum(hit_rate_model6) #very good predictive power: 0.909

#top-decile lift
#TDL table
decile_predicted_model6 <- ntile(predicted_model6, 10)
decile_model6 <- table(test_sample3$redeemer_latest_ind, decile_predicted_model6, dnn=
                         c("Observed", "Decile"))
decile_model6
#get the TDL
(decile_model6[2,10] / (decile_model6[1,10]+ decile_model6[2,10])) / mean(test_sample3$redeemer_latest_ind) #predicts 4.446 times better than random selection

#Gini coefficient
pred_model6 <- prediction(predicted_model6, test_sample3$redeemer_latest_ind)
perf_model6 <- performance(pred_model6,"tpr","fpr")
plot(perf_model6,xlab="Cumulative % of observations",ylab="Cumulative % of positive cases",xlim=c(0,1),ylim=c(0,1),xaxs="i",yaxs="i")
abline(0,1)
auc_model6 <- performance(pred_model6,"auc")
as.numeric(auc_model6@y.values)*2-1 #0.4: closer to 1 the better (if it is 0 that means no better predictions than random selection)

#Explain models ----
####Bagging_tree1
#useful for hypotheses testing
library("DALEX")
explanation_tree1 <- explain(
  model = Bagging_tree1,
  data = estimation_sample[, -10],
  y = estimation_sample$redeemer_latest_ind,,
  label = "Bagging_tree1"
)


pdp_bagging1_trips <- variable_effect(explanation_tree1, variable =  "trips_level", "partial_dependency")
plot(pdp_bagging1_trips)
pdp_bagging1_online <- variable_effect(explanation_tree1, variable =  "online_purch_level", "partial_dependency")
plot(pdp_bagging1_online)
pdp_bagging1_expenditure <- variable_effect(explanation_tree1, variable =  "expenditure_level", "partial_dependency")
plot(pdp_bagging1_expenditure)
pdp_bagging1_quantity <- variable_effect(explanation_tree1, variable =  "quantity_purchased_level", "partial_dependency")
plot(pdp_bagging1_quantity)
pdp_bagging1_last <- variable_effect(explanation_tree1, variable =  "last_expenditure_level", "partial_dependency")
plot(pdp_bagging1_last)

####Bagging_tree2
#only checking if variables behave similarly to the previous Bagging_tree1 (which was more complete with one extra variable)
explanation_tree2 <- explain(
  model = Bagging_tree2,
  data = estimation_sample[, -10],
  y = estimation_sample$redeemer_latest_ind,,
  label = "Bagging_tree2"
)


pdp_bagging2_trips <- variable_effect(explanation_tree2, variable =  "trips_level", "partial_dependency")
plot(pdp_bagging2_trips) #similarity confirmed
pdp_bagging2_expenditure <- variable_effect(explanation_tree2, variable =  "expenditure_level", "partial_dependency")
plot(pdp_bagging2_expenditure) #similarity confirmed
pdp_bagging2_quantity <- variable_effect(explanation_tree2, variable =  "quantity_purchased_level", "partial_dependency")
plot(pdp_bagging2_quantity) #similarity confirmed
pdp_bagging2_last <- variable_effect(explanation_tree2, variable =  "last_expenditure_level", "partial_dependency")
plot(pdp_bagging2_last) #similarity confirmed

#then, using Bagging_tree1 for hypotheses testing

####Bagging_tree3
#only checking if variables behave similarly to the previous Bagging_tree1 (which was more complete with one extra variable)
explanation_tree3 <- explain(
  model = Bagging_tree3,
  data = estimation_sample2[, -8],
  y = estimation_sample2$redeemer_latest_ind,,
  label = "Bagging_tree3"
)


pdp_bagging3_trips <- variable_effect(explanation_tree3, variable =  "trips_level", "partial_dependency")
plot(pdp_bagging3_trips) #similarity confirmed
pdp_bagging3_expenditure <- variable_effect(explanation_tree3, variable =  "expenditure_level", "partial_dependency")
plot(pdp_bagging3_expenditure) #somehow similar
pdp_bagging3_quantity <- variable_effect(explanation_tree3, variable =  "quantity_purchased_level", "partial_dependency")
plot(pdp_bagging3_quantity) #somehow similar
pdp_bagging3_last <- variable_effect(explanation_tree3, variable =  "last_expenditure_level", "partial_dependency")
plot(pdp_bagging3_last) #similarity confirmed

#then, using Bagging_tree1 for hypotheses testing

####Bagging_tree4
#important for hypothesis testing on pastLP_redemption variable. And checking if the other variables still behave similarly to Bagging_tree1
explanation_tree4 <- explain(
  model = Bagging_tree4,
  data = estimation_sample3[, -8],
  y = estimation_sample3$redeemer_latest_ind,,
  label = "Bagging_tree4"
)

pdp_bagging4_trips <- variable_effect(explanation_tree4, variable =  "trips_level", "partial_dependency")
plot(pdp_bagging4_trips) #somehow similar
pdp_bagging4_expenditure <- variable_effect(explanation_tree4, variable =  "expenditure_level", "partial_dependency")
plot(pdp_bagging4_expenditure) #similarity confirmed
pdp_bagging4_quantity <- variable_effect(explanation_tree4, variable =  "quantity_purchased_level", "partial_dependency")
plot(pdp_bagging4_quantity) #similarity confirmed
pdp_bagging4_last <- variable_effect(explanation_tree4, variable =  "last_expenditure_level", "partial_dependency")
plot(pdp_bagging4_last) #similarity confirmed
pdp_bagging4_past <- variable_effect(explanation_tree4, variable =  "pastLP_redemption", "partial_dependency")
plot(pdp_bagging4_past)
#plotting pdp_bagging4_past with bars
x_values <- pdp_bagging4_past[, 3]
y_values <- pdp_bagging4_past[, 4]
barplot(y_values, names.arg = x_values, col = c("grey", "navy"), beside = TRUE, main = "Partial dependency bar plot",
        xlab = "pastLP_redemption", ylab = "redemption likelihood", ylim = c(0.0, 0.3)) #interesting: relatively strong postive influence of pastLP_redemption on redemption likelihood

#then, using Bagging_tree1 for hypotheses testing (all vars) + Bagging_tree4 for hypothesis testing (pastLP_redemption var)

#plotting TDL of two tranfer models and last model
plot(decile_model3[2,], type = "o", col = "blue", pch = 1, ylim = c(0, 10000), xlab = "Deciles", ylab = "Redeemers")
lines(decile_model5[2,], type = "o", col = "red", pch = 2)
#adding a legend
legend("topright", legend = c("Bagging II - Transfer Across Retailers", "Bagging III - Transfer on Same Retailer"), col = c("blue", "red"), pch = 1:3, bty = "n")
