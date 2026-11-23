# ==============================================================================
# WASTE DETECTION — FULL PIPELINE (CORRELATION FILTERED) + 32 MODELS
# ==============================================================================

library(terra)
library(sf)
library(dplyr)
library(stringr)
library(tidyr)
library(caret)
library(rsample)
library(ggplot2)
library(flextable)
library(officer)
library(tictoc)
library(effectsize)
library(emmeans)
library(broom)

#===============================================================================
# 1. DATA LOADING
#===============================================================================

rgb_dir   <- "data/tiled_data/images/"
multi_dir <- "data/tiled_data/multiband/"
mask_dir  <- "data/tiled_data/masks_bin/"

get_tile_id <- function(path) {
  fname <- basename(path)
  str_replace_all(fname, "clipped_image_|mask_|_combined|\\.tif", "")
}

files_df <- data.frame(rgb_path = list.files(rgb_dir, pattern="\\.tif$", full.names=TRUE)) %>%
  mutate(id = get_tile_id(rgb_path)) %>%
  inner_join(
    data.frame(multi_path = list.files(multi_dir, pattern="combined\\.tif$", full.names=TRUE)) %>%
      mutate(id = get_tile_id(multi_path)),
    by="id"
  ) %>%
  inner_join(
    data.frame(mask_path = list.files(mask_dir, pattern=".*\\.tif$", full.names=TRUE)) %>%
      mutate(id = get_tile_id(mask_path)),
    by="id"
  )

texture_names <- c("Energy","Entropy","Cor","IDM","Inertia","CS","CP","HC",
                   "Mean","Variance","Dissimilarity","SumAverage","SumVariance",
                   "SumEntropy","Dif_of_Entropies","Dif_of_Variances","IC1","IC2",
                   "SRE","LRE","GLN","RLN","RP","LGRE","HGLRE",
                   "SRLGLE","SRHGLE","LRLGLE")

image_list <- lapply(1:nrow(files_df), function(i) {
  row <- files_df[i,]
  mask_r <- rast(row$mask_path)
  rgb_r  <- rast(row$rgb_path)[[1:3]]
  tex_r  <- rast(row$multi_path)
  
  names(rgb_r) <- c("Red","Green","Blue")
  names(tex_r) <- texture_names[1:nlyr(tex_r)]
  names(mask_r) <- "waste_id"
  
  list(mask=mask_r, optical=rgb_r, texture=tex_r, id=row$id)
})

#===============================================================================
# 2. DATAFRAMES: PIXEL + OBJECT
#===============================================================================

pixel_df <- lapply(image_list, function(img) {
  as.data.frame(c(img$optical, img$texture, img$mask), na.rm=TRUE)
}) %>% bind_rows() %>%
  mutate(
    class_bin   = factor(ifelse(waste_id>0,"Waste","NonWaste"), levels=c("NonWaste","Waste")),
    class_multi = as.factor(waste_id)
  )

object_df <- lapply(image_list, function(img) {
  polys <- as.polygons(img$mask, dissolve=TRUE)
  stats <- terra::extract(c(img$optical, img$texture), polys, fun=mean, na.rm=TRUE)
  
  df_out <- data.frame(values(polys), stats) %>%
    select(-ID) %>%
    rename(waste_id = 1) %>%
    mutate(
      class_bin   = factor(ifelse(waste_id>0,"Waste","NonWaste"), levels=c("NonWaste","Waste")),
      class_multi = as.factor(waste_id)
    )
  df_out
}) %>% bind_rows()

#===============================================================================
# 3. PIXEL SAMPLING & SPLITS
#===============================================================================

set.seed(42)
sample_size <- 5000
n_waste    <- sum(pixel_df$class_bin=="Waste")
n_nonwaste <- sum(pixel_df$class_bin=="NonWaste")
per_class  <- min(sample_size/2, n_waste, n_nonwaste)

pixel_sampled <- pixel_df %>%
  group_by(class_bin) %>%
  sample_n(size=per_class) %>%
  ungroup()

set.seed(42)
pix_split <- initial_split(pixel_sampled, prop=0.80, strata=class_bin)
train_data_bin   <- training(pix_split)
test_data_bin    <- testing(pix_split)

set.seed(42)
obj_split <- initial_split(object_df, prop=0.80, strata=class_bin)
obj_train_bin   <- training(obj_split)
obj_test_bin    <- testing(obj_split)

#===============================================================================
# 4. FEATURE ELIMINATION (CORRELATION FILTER)
#===============================================================================

optical_vars <- c("Red","Green","Blue")
texture_vars <- texture_names # All potential textures

# Calculate correlation on pixel training data
cor_matrix <- cor(train_data_bin[, texture_vars], use = "complete.obs")

# Identify variables with correlation > 0.90 to eliminate redundancy
high_cor_idx <- findCorrelation(cor_matrix, cutoff = 0.90)
excluded_vars <- texture_vars[high_cor_idx]
reduced_texture_vars <- texture_vars[-high_cor_idx]

# Combine for the final "Texture" model feature set
reduced_vars <- c(optical_vars, reduced_texture_vars)

cat("--- FEATURE SELECTION SUMMARY ---\n")
cat("Excluded (correlated):", paste(excluded_vars, collapse=", "), "\n")
cat("Final Texture Variable Count:", length(reduced_vars), "\n\n")

#===============================================================================
# 5. TRAIN FUNCTIONS
#===============================================================================

build_formula <- function(resp, vars) {
  as.formula(paste(resp, "~", paste(vars, collapse=" + ")))
}

train_bin <- function(train, test, vars, method) {
  f <- build_formula("class_bin", vars)
  tic()
  model <- train(f, data=train, method=method,
                 trControl=trainControl(method="repeatedcv", number=5, repeats=5),
                 verbose=FALSE)
  toc()
  test$pred <- predict(model, test)
  list(model=model, confusion=confusionMatrix(test$pred, test$class_bin, mode="everything"))
}

train_multi <- function(train, test, vars, method) {
  f <- build_formula("class_multi", vars)
  tic()
  model <- train(f, data=train, method=method,
                 trControl=trainControl(method="repeatedcv", number=5, repeats=5),
                 verbose=FALSE)
  toc()
  test$pred <- predict(model, test)
  list(model=model, confusion=confusionMatrix(test$pred, test$class_multi, mode="everything"))
}

#===============================================================================
# 6. TRAIN ALL 32 MODELS
#===============================================================================

# Binary Models (Pixel & Object)
rf_bin_opt_pix  <- train_bin(train_data_bin, test_data_bin, optical_vars, "rf")
ann_bin_opt_pix <- train_bin(train_data_bin, test_data_bin, optical_vars, "nnet")
nb_bin_opt_pix  <- train_bin(train_data_bin, test_data_bin, optical_vars, "nb")
svm_bin_opt_pix <- train_bin(train_data_bin, test_data_bin, optical_vars, "svmLinear")

rf_bin_tex_pix  <- train_bin(train_data_bin, test_data_bin, reduced_vars, "rf")
ann_bin_tex_pix <- train_bin(train_data_bin, test_data_bin, reduced_vars, "nnet")
nb_bin_tex_pix  <- train_bin(train_data_bin, test_data_bin, reduced_vars, "nb")
svm_bin_tex_pix <- train_bin(train_data_bin, test_data_bin, reduced_vars, "svmLinear")

rf_bin_opt_obj  <- train_bin(obj_train_bin, obj_test_bin, optical_vars, "rf")
ann_bin_opt_obj <- train_bin(obj_train_bin, obj_test_bin, optical_vars, "nnet")
nb_bin_opt_obj  <- train_bin(obj_train_bin, obj_test_bin, optical_vars, "nb")
svm_bin_opt_obj <- train_bin(obj_train_bin, obj_test_bin, optical_vars, "svmLinear")

rf_bin_tex_obj  <- train_bin(obj_train_bin, obj_test_bin, reduced_vars, "rf")
ann_bin_tex_obj <- train_bin(obj_train_bin, obj_test_bin, reduced_vars, "nnet")
nb_bin_tex_obj  <- train_bin(obj_train_bin, obj_test_bin, reduced_vars, "nb")
svm_bin_tex_obj <- train_bin(obj_train_bin, obj_test_bin, reduced_vars, "svmLinear")

# Multiclass Models (Pixel & Object)
rf_multi_opt_pix  <- train_multi(train_data_bin, test_data_bin, optical_vars, "rf")
ann_multi_opt_pix <- train_multi(train_data_bin, test_data_bin, optical_vars, "nnet")
nb_multi_opt_pix  <- train_multi(train_data_bin, test_data_bin, optical_vars, "nb")
svm_multi_opt_pix <- train_multi(train_data_bin, test_data_bin, optical_vars, "svmLinear")

rf_multi_tex_pix  <- train_multi(train_data_bin, test_data_bin, reduced_vars, "rf")
ann_multi_tex_pix <- train_multi(train_data_bin, test_data_bin, reduced_vars, "nnet")
nb_multi_tex_pix  <- train_multi(train_data_bin, test_data_bin, reduced_vars, "nb")
svm_multi_tex_pix <- train_multi(train_data_bin, test_data_bin, reduced_vars, "svmLinear")

rf_multi_opt_obj  <- train_multi(obj_train_bin, obj_test_bin, optical_vars, "rf")
ann_multi_opt_obj <- train_multi(obj_train_bin, obj_test_bin, optical_vars, "nnet")
nb_multi_opt_obj  <- train_multi(obj_train_bin, obj_test_bin, optical_vars, "nb")
svm_multi_opt_obj <- train_multi(obj_train_bin, obj_test_bin, optical_vars, "svmLinear")

rf_multi_tex_obj  <- train_multi(obj_train_bin, obj_test_bin, reduced_vars, "rf")
ann_multi_tex_obj <- train_multi(obj_train_bin, obj_test_bin, reduced_vars, "nnet")
nb_multi_tex_obj  <- train_multi(obj_train_bin, obj_test_bin, reduced_vars, "nb")
svm_multi_tex_obj <- train_multi(obj_train_bin, obj_test_bin, reduced_vars, "svmLinear")

#===============================================================================
# 7. SUMMARY & EXPORT
#===============================================================================

extract_scores <- function(model_obj, model_name) {
  cm <- model_obj$confusion
  byclass <- cm$byClass
  if (is.matrix(byclass)) {
    f1 <- mean(byclass[,"F1"], na.rm=TRUE)
    prec <- mean(byclass[,"Precision"], na.rm=TRUE)
    rec <- mean(byclass[,"Recall"], na.rm=TRUE)
  } else {
    f1 <- byclass["F1"]; prec <- byclass["Precision"]; rec <- byclass["Recall"]
  }
  data.frame(Model = model_name, Precision = round(prec, 4), Recall = round(rec, 4), F1 = round(f1, 4))
}

all_models <- list(
  "Pixel Bin – RF Optical" = rf_bin_opt_pix, "Pixel Bin – ANN Optical" = ann_bin_opt_pix, 
  "Pixel Bin – NB Optical" = nb_bin_opt_pix, "Pixel Bin – SVM Optical" = svm_bin_opt_pix,
  "Pixel Bin – RF Texture" = rf_bin_tex_pix, "Pixel Bin – ANN Texture" = ann_bin_tex_pix,
  "Pixel Bin – NB Texture" = nb_bin_tex_pix, "Pixel Bin – SVM Texture" = svm_bin_tex_pix,
  "Object Bin – RF Optical" = rf_bin_opt_obj, "Object Bin – ANN Optical" = ann_bin_opt_obj,
  "Object Bin – NB Optical" = nb_bin_opt_obj, "Object Bin – SVM Optical" = svm_bin_opt_obj,
  "Object Bin – RF Texture" = rf_bin_tex_obj, "Object Bin – ANN Texture" = ann_bin_tex_obj,
  "Object Bin – NB Texture" = nb_bin_tex_obj, "Object Bin – SVM Texture" = svm_bin_tex_obj,
  "Pixel Multi – RF Optical" = rf_multi_opt_pix, "Pixel Multi – ANN Optical" = ann_multi_opt_pix,
  "Pixel Multi – NB Optical" = nb_multi_opt_pix, "Pixel Multi – SVM Optical" = svm_multi_opt_pix,
  "Pixel Multi – RF Texture" = rf_multi_tex_pix, "Pixel Multi – ANN Texture" = ann_multi_tex_pix,
  "Pixel Multi – NB Texture" = nb_multi_tex_pix, "Pixel Multi – SVM Texture" = svm_multi_tex_pix,
  "Object Multi – RF Optical" = rf_multi_opt_obj, "Object Multi – ANN Optical" = ann_multi_opt_obj,
  "Object Multi – NB Optical" = nb_multi_opt_obj, "Object Multi – SVM Optical" = svm_multi_opt_obj,
  "Object Multi – RF Texture" = rf_multi_tex_obj, "Object Multi – ANN Texture" = ann_multi_tex_obj,
  "Object Multi – NB Texture" = nb_multi_tex_obj, "Object Multi – SVM Texture" = svm_multi_tex_obj
)

summary_table <- do.call(rbind, lapply(names(all_models), function(nm) extract_scores(all_models[[nm]], nm)))

# Word Export
summary_ft <- flextable(summary_table) %>%
  theme_booktabs() %>% autofit() %>%
  font(fontname = "Times New Roman", part = "all")

doc <- read_docx() %>%
  body_add_par("Model Performance Summary", style = "heading 1") %>%
  body_add_flextable(summary_ft)
print(doc, target = "Model_Performance_Summary.docx")

#===============================================================================
# 8. STATISTICAL ANALYSIS (ANOVA & EFFECT SIZE)
#===============================================================================

df_stats <- summary_table %>%
  separate(Model, into = c("DataModel","ClassType","Algorithm","FeatureType"), sep = " – | ") %>%
  mutate(across(everything(), factor))

lm_main <- lm(F1 ~ DataModel + ClassType + FeatureType + Algorithm, data = df_stats)
eta_res  <- eta_squared(lm_main, partial = TRUE)

# Variance Plot
ggplot(eta_res, aes(x = reorder(Parameter, Eta2_partial), y = Eta2_partial)) +
  geom_col(fill = "steelblue") + coord_flip() + theme_minimal() +
  labs(x = "Factor", y = "Eta^2 (Partial)", title = "Relative Importance on F1")

# Marginal Means
emm_algo <- emmeans(lm_main, ~ Algorithm)
plot(emm_algo) + theme_minimal() + labs(title = "Estimated F1 by Algorithm")

cat("✅ Complete pipeline finished. Outputs saved.\n")