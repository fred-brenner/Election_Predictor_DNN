library(data.table)
library(parallel)

# function plurality_rule
# input: a non-negative integer number of voters,
#        a vector with probability mass function for options,
#        and the options index
# output: probability that first option wins under plurality rule
plurality_rule <- function(total.voters, options, options_index) {
  num.options <- length(options)
  
  # Calculating how many votes are needed to win
  ## Standard. Comment for two options
  winning <- ceiling(total.voters / num.options) + 1
  
  ## Un-comment for two options
  # winning = ceiling((total.voters + 1) / num.options)
  
  difference <- total.voters - winning
  
  df <- RcppAlgos::compositionsGeneral(0:difference, length(options), repetition = TRUE, weak = TRUE)
  
  df[, 1] <- df[, 1] + winning
  
  # Filtering cases in which the first option is the winning options with a general algorithm
  
  ## Standard. Comment and choose one below to improve speed
  # preferred = c()
  # for (option in 2:length(options)) {
  #    most_a = which(df[,1] > df[,option])
  #    if (option == 2) {
  #      preferred = c(preferred, most_a)
  #    } else {
  #      preferred = intersect(preferred, most_a)
  #    }
  #  }
  
  ########################################################################
  # Filtering cases in which the first option is the winning options with a specific algorithm
  
  # Un-comment for two options
  # preferred = which(df[,1] > df[,2])
  
  # Un-comment for three options
  # preferred = intersect(which(df[,1] > df[,2]), which(df[,1] > df[,3]))
  
  # Un-comment for four options
  preferred_b <- intersect(which(df[, 1] > df[, 2]), which(df[, 1] > df[, 3]))
  preferred <- intersect(preferred_b, which(df[, 1] > df[, 4]))
  
  # Un-comment for five options
  # preferred_b = intersect(which(df[,1] > df[,2]), which(df[,1] > df[,3]))
  # preffered_c = intersect(which(df[,1] > df[,4]), which(df[,1] > df[,5]))
  # preferred = intersect(preferred_b, preffered_c)
  
  ########################################################################
  
  # Choosing only the cases where the first option wins
  df <- df[preferred, ]
  
  df <- as.data.frame(df)
  rows <- nrow(df)
  
  # Convert the data.frame to a data.table
  table <- data.table::data.table(df)
  
  # Put the data in long format
  table <- data.table::melt(table, measure.vars = names(df))
  
  table[, group := rep(1:rows, num.options)]
  
  # Apply function to each group
  table[, probability := stats::dmultinom(value, prob = options[options_index])]
  
  whole.group <- head(table, rows)
  prob <- sum(whole.group$probability)
  
  return(prob)
}

# #Example to run the function
# 
# ##voters
# total.voters <- 1001
# ##options
# options <- c(0.26, 0.25, 0.25, 0.24)
# ## one call of the function
# plurality_rule(total.voters, options, 1)

# Exporting data
voters_nn_odd <- seq(12, 2012, 4)
options <- c(0.3, 0.25, 0.25, 0.2)
p_nn_odd <- c()

# Create a cluster with 3 processes
cl <- parallel::makeCluster(3)

# Export the necessary functions and packages to the worker processes
parallel::clusterExport(cl, c("plurality_rule", "options", "data.table", "melt", "RcppAlgos"))

# Apply the function in parallel using the cluster
p_nn_odd <- parallel::parLapply(cl, voters_nn_odd, function(i) plurality_rule(i, options, 1))

# Stop the cluster
parallel::stopCluster(cl)

df_odd <- data.frame(probability_first_option_wins = rep(options[1], length(voters_nn_odd)), size = voters_nn_odd, probability = p_nn_odd)
write.csv(df_odd, "0.3, 0.25, 0.25, 0.2 #12-1012 (only multiples of 4).csv")
