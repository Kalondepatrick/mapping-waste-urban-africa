# Read the file and assign it to a variable
my_data <- readRDS("data/trained_models/geobia_r/RF.rds")
my_data <- readRDS("data/trained_models/geobia_r/NB.rds")
my_data <- readRDS("data/trained_models/geobia_r/ANN.rds")
my_data <- readRDS("data/trained_models/geobia_r/SVM.rds")
# View the first few rows
head(my_data)
