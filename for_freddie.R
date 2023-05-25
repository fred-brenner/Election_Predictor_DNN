
library(RcppAlgos)
library(data.table)

# function plurality_rule
# input: a non negative integer number of voters and
#        a vector with probability mass function for options
# output: probability that first option wins under plurality rule

plurality_rule <- function (total.voters, options) { 
  
  num.options = length(options)
  
  # Calculating how many votes are neede to win
  ## Standard. Comment for two options
  winning = ceiling(total.voters/num.options) + 1
  
  ## Un-commment for two options
  #winning = ceiling((total.voters + 1)/num.options)
  
  difference = total.voters-winning
  
  df = compositionsGeneral(0:difference, length(options), repetition = TRUE, weak = TRUE)
  
  df[,1] = df[,1]+winning
  
  #Filtering cases in which the first option is the winning options with a general algorithm 
  
  ## Standard. Comment and choose one below to improve speed
  #preferred = c()
  #for (option in 2:length(options)) {
  #   most_a = which(df[,1] > df[,option])
  #   if (option == 2) {
  #     preferred = c(preferred, most_a)
  #   } else {
  #     preferred = intersect(preferred, most_a)
  #   }
  # }
  
  ########################################################################
  #Filtering cases in which the first option is the winning options with an specific algorithm 
  
  #Un-comment for two options
  #preferred = which(df[,1] > df[,2])
  
  #Un-comment for three options
  #preferred = intersect(which(df[,1] > df[,2]), which(df[,1] > df[,3]))
  
  #Un-comment for four options
  preferred_b = intersect(which(df[,1] > df[,2]), which(df[,1] > df[,3]))
  preferred = intersect(preferred_b, which(df[,1] > df[,4]))
  
  #Un-comment for five options
  # preferred_b = intersect(which(df[,1] > df[,2]), which(df[,1] > df[,3]))
  # preffered_c = intersect(which(df[,1] > df[,4]), which(df[,1] > df[,5]))
  # preferred = intersect(preferred_b, preffered_c)
  
  ########################################################################
  
  
  # Choosing only the cases where the first option wins
  df = df[preferred,]
  
  df=as.data.frame(df)
  rows = nrow(df)
  
  #convert the data.frame to a data.table
  table = setDT(df)
  
  #put the data in long format
  table = data.table::melt(df, measure.vars = names(df))
  
  table[, group := rep(1:rows, num.options)]
  
  #apply function to each group
  table[, probability := dmultinom(value, prob = options), by = "group"]
  
  whole.group = head(table, rows)
  prob = sum(whole.group$probability)
  
  return(prob)
  
}

# #Example to run the function
# 
# ##voters
# total.voters = 1001
# ##options
# options = c(0.26, 0.25, 0.25, 0.24)
# ## one call of the function 
# plurality_rule(total.voters, options)

# Exporting data
voters_nn_odd = seq(12, 1612, 16)
options = c(0.3, 0.25, 0.25, 0.2)
p_nn_odd = c()
for (i in voters_nn_odd){
  print(i)
  p_nn_odd <- c(p_nn_odd, plurality_rule(i , options) )
}

df_odd = data.frame(probability_first_option_wins = rep(options[1], length(voters_nn_odd)), size=voters_nn_odd, probability=p_nn_odd)
write.csv(df_odd, "0.3, 0.25, 0.25, 0.2 #12-1012 (only multiples of 4).csv")