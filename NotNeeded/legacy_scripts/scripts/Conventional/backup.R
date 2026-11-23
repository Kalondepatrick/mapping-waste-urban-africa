# ==============================================================================
# WASTE DETECTION — FULL PIPELINE + 32 MODELS + WORD EXPORT
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
# 2b. PIXEL SAMPLING (BALANCED 5000)
#===============================================================================

set.seed(42)
sample_size <- 5000

n_waste    <- sum(pixel_df$class_bin=="Waste")
n_nonwaste <- sum(pixel_df$class_bin=="NonWaste")

per_class <- min(sample_size/2, n_waste, n_nonwaste)

pixel_sampled <- pixel_df %>%
  group_by(class_bin) %>%
  sample_n(size=per_class) %>%
  ungroup()

cat("✅ Sampled pixels:", nrow(pixel_sampled), "\n")

#===============================================================================
# 3. TRAIN/TEST SPLITS
#===============================================================================

set.seed(42)
pix_split <- initial_split(pixel_sampled, prop=0.80, strata=class_bin)

train_data_bin   <- training(pix_split)
test_data_bin    <- testing(pix_split)
train_data_multi <- train_data_bin
test_data_multi  <- test_data_bin

set.seed(42)
obj_split <- initial_split(object_df, prop=0.80, strata=class_bin)

obj_train_bin   <- training(obj_split)
obj_test_bin    <- testing(obj_split)
obj_train_multi <- obj_train_bin
obj_test_multi  <- obj_test_bin

#===============================================================================
# 4. MODEL FEATURES + TRAIN FUNCTIONS
#===============================================================================

optical_vars <- c("Red","Green","Blue")
full_vars <- c(
  "Red","Green","Blue",
  "Energy","Entropy","Cor","IDM","Inertia","CS","CP","HC",
  "Mean","Variance","Dissimilarity","SumAverage","SumVariance",
  "SumEntropy","Dif_of_Entropies","Dif_of_Variances","IC1","IC2",
  "SRE","LRE","GLN","RLN","RP","LGRE","HGLRE","SRLGLE","SRHGLE","LRLGLE"
)

build_formula <- function(resp, vars) {
  as.formula(paste(resp, "~", paste(vars, collapse=" + ")))
}

train_bin <- function(train, test, vars, method) {
  f <- build_formula("class_bin", vars)
  tic()
  model <- train(
    f, data=train, method=method,
    trControl=trainControl(method="repeatedcv", number=5, repeats=5),
    verbose=FALSE
  )
  toc()
  test$pred <- predict(model, test)
  confusion <- confusionMatrix(test$pred, test$class_bin, mode="everything")
  list(model=model, confusion=confusion)
}

train_multi <- function(train, test, vars, method) {
  f <- build_formula("class_multi", vars)
  tic()
  model <- train(
    f, data=train, method=method,
    trControl=trainControl(method="repeatedcv", number=5, repeats=5),
    verbose=FALSE
  )
  toc()
  test$pred <- predict(model, test)
  confusion <- confusionMatrix(test$pred, test$class_multi, mode="everything")
  list(model=model, confusion=confusion)
}

#===============================================================================
# 5. TRAIN ALL 32 MODELS
#===============================================================================

methods <- c("rf","nnet","nb","svmLinear")

# Pixel Binary — Optical
rf_bin_opt_pix  <- train_bin(train_data_bin, test_data_bin, optical_vars, "rf")
ann_bin_opt_pix <- train_bin(train_data_bin, test_data_bin, optical_vars, "nnet")
nb_bin_opt_pix  <- train_bin(train_data_bin, test_data_bin, optical_vars, "nb")
svm_bin_opt_pix <- train_bin(train_data_bin, test_data_bin, optical_vars, "svmLinear")

# Pixel Binary — Texture
rf_bin_tex_pix  <- train_bin(train_data_bin, test_data_bin, full_vars, "rf")
ann_bin_tex_pix <- train_bin(train_data_bin, test_data_bin, full_vars, "nnet")
nb_bin_tex_pix  <- train_bin(train_data_bin, test_data_bin, full_vars, "nb")
svm_bin_tex_pix <- train_bin(train_data_bin, test_data_bin, full_vars, "svmLinear")

# Object Binary — Optical
rf_bin_opt_obj  <- train_bin(obj_train_bin, obj_test_bin, optical_vars, "rf")
ann_bin_opt_obj <- train_bin(obj_train_bin, obj_test_bin, optical_vars, "nnet")
nb_bin_opt_obj  <- train_bin(obj_train_bin, obj_test_bin, optical_vars, "nb")
svm_bin_opt_obj <- train_bin(obj_train_bin, obj_test_bin, optical_vars, "svmLinear")

# Object Binary — Texture
rf_bin_tex_obj  <- train_bin(obj_train_bin, obj_test_bin, full_vars, "rf")
ann_bin_tex_obj <- train_bin(obj_train_bin, obj_test_bin, full_vars, "nnet")
nb_bin_tex_obj  <- train_bin(obj_train_bin, obj_test_bin, full_vars, "nb")
svm_bin_tex_obj <- train_bin(obj_train_bin, obj_test_bin, full_vars, "svmLinear")

# Pixel Multiclass — Optical
rf_multi_opt_pix  <- train_multi(train_data_multi, test_data_multi, optical_vars, "rf")
ann_multi_opt_pix <- train_multi(train_data_multi, test_data_multi, optical_vars, "nnet")
nb_multi_opt_pix  <- train_multi(train_data_multi, test_data_multi, optical_vars, "nb")
svm_multi_opt_pix <- train_multi(train_data_multi, test_data_multi, optical_vars, "svmLinear")

# Pixel Multiclass — Texture
rf_multi_tex_pix  <- train_multi(train_data_multi, test_data_multi, full_vars, "rf")
ann_multi_tex_pix <- train_multi(train_data_multi, test_data_multi, full_vars, "nnet")
nb_multi_tex_pix  <- train_multi(train_data_multi, test_data_multi, full_vars, "nb")
svm_multi_tex_pix <- train_multi(train_data_multi, test_data_multi, full_vars, "svmLinear")

# Object Multiclass — Optical
rf_multi_opt_obj  <- train_multi(obj_train_multi, obj_test_multi, optical_vars, "rf")
ann_multi_opt_obj <- train_multi(obj_train_multi, obj_test_multi, optical_vars, "nnet")
nb_multi_opt_obj  <- train_multi(obj_train_multi, obj_test_multi, optical_vars, "nb")
svm_multi_opt_obj <- train_multi(obj_train_multi, obj_test_multi, optical_vars, "svmLinear")

# Object Multiclass — Texture
rf_multi_tex_obj  <- train_multi(obj_train_multi, obj_test_multi, full_vars, "rf")
ann_multi_tex_obj <- train_multi(obj_train_multi, obj_test_multi, full_vars, "nnet")
nb_multi_tex_obj  <- train_multi(obj_train_multi, obj_test_multi, full_vars, "nb")
svm_multi_tex_obj <- train_multi(obj_train_multi, obj_test_multi, full_vars, "svmLinear")

#===============================================================================
# 6. RESULTS → WORD DOCUMENT
#===============================================================================




# =====================================================================
# AUTOMATIC SUMMARY TABLE FOR ALL 32 MODELS
# =====================================================================

extract_scores <- function(model_obj, model_name) {
  cm <- model_obj$confusion
  
  
  # For binary: use Positive class metrics
  # For multiclass: macro average F1 from caret
  byclass <- cm$byClass
  
  # Handle multi-class (caret returns a matrix)
  if (is.matrix(byclass)) {
    precision <- mean(byclass[,"Precision"], na.rm=TRUE)
    recall    <- mean(byclass[,"Recall"], na.rm=TRUE)
    f1        <- mean(byclass[,"F1"], na.rm=TRUE)
  } else {
    precision <- byclass["Precision"]
    recall    <- byclass["Recall"]
    f1        <- byclass["F1"]
  }
  
  data.frame(
    Model = model_name,
    Precision = round(precision, 4),
    Recall    = round(recall, 4),
    F1        = round(f1, 4)
  )
}

# List ALL models in a single named list
all_models <- list(
  "Pixel Bin – RF Optical"  = rf_bin_opt_pix,
  "Pixel Bin – ANN Optical" = ann_bin_opt_pix,
  "Pixel Bin – NB Optical"  = nb_bin_opt_pix,
  "Pixel Bin – SVM Optical" = svm_bin_opt_pix,
  
  "Pixel Bin – RF Texture"  = rf_bin_tex_pix,
  "Pixel Bin – ANN Texture" = ann_bin_tex_pix,
  "Pixel Bin – NB Texture"  = nb_bin_tex_pix,
  "Pixel Bin – SVM Texture" = svm_bin_tex_pix,
  
  "Object Bin – RF Optical" = rf_bin_opt_obj,
  "Object Bin – ANN Optical"= ann_bin_opt_obj,
  "Object Bin – NB Optical" = nb_bin_opt_obj,
  "Object Bin – SVM Optical"= svm_bin_opt_obj,
  
  "Object Bin – RF Texture" = rf_bin_tex_obj,
  "Object Bin – ANN Texture"= ann_bin_tex_obj,
  "Object Bin – NB Texture" = nb_bin_tex_obj,
  "Object Bin – SVM Texture"= svm_bin_tex_obj,
  
  "Pixel Multi – RF Optical"  = rf_multi_opt_pix,
  "Pixel Multi – ANN Optical" = ann_multi_opt_pix,
  "Pixel Multi – NB Optical"  = nb_multi_opt_pix,
  "Pixel Multi – SVM Optical" = svm_multi_opt_pix,
  
  "Pixel Multi – RF Texture"  = rf_multi_tex_pix,
  "Pixel Multi – ANN Texture" = ann_multi_tex_pix,
  "Pixel Multi – NB Texture"  = nb_multi_tex_pix,
  "Pixel Multi – SVM Texture" = svm_multi_tex_pix,
  
  "Object Multi – RF Optical" = rf_multi_opt_obj,
  "Object Multi – ANN Optical"= ann_multi_opt_obj,
  "Object Multi – NB Optical" = nb_multi_opt_obj,
  "Object Multi – SVM Optical"= svm_multi_opt_obj,
  
  "Object Multi – RF Texture" = rf_multi_tex_obj,
  "Object Multi – ANN Texture"= ann_multi_tex_obj,
  "Object Multi – NB Texture" = nb_multi_tex_obj,
  "Object Multi – SVM Texture"= svm_multi_tex_obj
)

# Build a full summary table
summary_table <- do.call(
  rbind,
  lapply(names(all_models), function(nm) {
    extract_scores(all_models[[nm]], nm)
  })
)

print(summary_table)







# Professional journal-style theme
format_journal_table <- function(ft) {
  ft %>%
    theme_booktabs() %>%                       # clean professional borders
    align(align = "center", part = "all") %>%  # center columns
    bold(part = "header") %>%                  # bold header
    fontsize(size = 11, part = "all") %>%      # uniform font size
    font(fontname = "Times New Roman", part = "all") %>% 
    padding(padding = 6, part = "all") %>%     # more breathing room
    autofit()                                  # adjust column width
}

# Create summary table for Word
summary_ft <- flextable(summary_table)
summary_ft <- format_journal_table(summary_ft)

# Build Word document
doc <- read_docx() %>%
  body_add_par("Model Performance Summary (All 32 Models)", 
               style = "heading 1") %>%
  body_add_par(" ", style = NULL) %>%   # spacing
  body_add_flextable(summary_ft)

print(doc, target = "Model_Performance_Summary.docx")

cat("✅ Summary table saved to Model_Performance_Summary.docx\n")





# =====================================================
# 1. Load libraries
# =====================================================

library(tidyverse)
library(effectsize)
library(emmeans)
library(ggplot2)

# =====================================================
# 2. Split factorial variables (your starting point)
# =====================================================

df <- summary_table %>%
  separate(Model,
           into = c("DataModel","ClassType","Algorithm","FeatureType"),
           sep = " – | ")

# =====================================================
# 3. Convert to factors
# =====================================================

df <- df %>%
  mutate(
    DataModel  = factor(DataModel, levels = c("Pixel","Object")),
    ClassType  = factor(ClassType, levels = c("Bin","Multi")),
    Algorithm  = factor(Algorithm),
    FeatureType = factor(FeatureType, levels = c("Optical","Texture"))
  )

str(df)

# =====================================================


############################################################
# 1. LIBRARIES
############################################################

library(tidyverse)
library(effectsize)
library(emmeans)
library(broom)

############################################################
# 2. CHECK DESIGN (optional sanity check)
############################################################

table(df$DataModel,
      df$ClassType,
      df$FeatureType,
      df$Algorithm)
# should show all cells = 1

############################################################
# 3. FIT EFFECT-ESTIMATION MODEL
# (NO high-order interactions)
############################################################

lm_main <- lm(
  F1 ~ DataModel + ClassType + FeatureType + Algorithm,
  data = df
)

summary(lm_main)
anova(lm_main)

############################################################
# 4. VARIANCE DECOMPOSITION (KEY RESULT)
############################################################

eta_res <- eta_squared(lm_main, partial = TRUE)

eta_res



############################################################
# Variance contribution plot
############################################################

eta_res %>%
  ggplot(aes(x = reorder(Parameter, Eta2_partial),
             y = Eta2_partial)) +
  geom_col() +
  coord_flip() +
  theme_minimal(base_size = 14) +
  labs(
    x = "",
    y = expression("Explained variance ("*eta^2*")"),
    title = "Relative Importance of Modelling Decisions on F1 Score"
  )


############################################################
# Marginal means (average performance per factor)
############################################################

emm_data_model <- emmeans(lm_main, ~ DataModel)
emm_algorithm  <- emmeans(lm_main, ~ Algorithm)
emm_feature    <- emmeans(lm_main, ~ FeatureType)
emm_class      <- emmeans(lm_main, ~ ClassType)

emm_data_model
emm_algorithm
emm_feature
emm_class



pairs(emm_algorithm)



############################################################
# Algorithm performance plot
############################################################

emm_algorithm %>%
  as.data.frame() %>%
  ggplot(aes(Algorithm, emmean)) +
  geom_point(size = 4) +
  geom_errorbar(aes(ymin = lower.CL,
                    ymax = upper.CL),
                width = 0.2) +
  theme_minimal(base_size = 14) +
  labs(
    y = "Estimated Mean F1 Score",
    title = "Average Algorithm Performance"
  )



############################################################
# Clean coefficient table
############################################################

tidy(lm_main, conf.int = TRUE) %>%
  arrange(desc(abs(estimate)))



