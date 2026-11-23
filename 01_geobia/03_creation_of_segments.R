library(terra)
library(sf)
library(dplyr)
library(stringr)


library(raster)
library(sf)
library(parallel)
library(doParallel)

image_folder <- "data/tiled_data/partitioned/modelDevelopment/images/"
output_base  <- "data/tiled_data/partitioned/modelDevelopment/segments/"


if (!dir.exists(output_base)) {
  dir.create(output_base, recursive = TRUE)
}


tif_files <- list.files(image_folder, pattern = "\\.tif$", full.names = TRUE)


spatial_radius <- 50
range_radius   <- 30
otb_path       <- "C:/Users/Peter/Documents/Tools/OTB-9.1.0-Win64/bin/otbcli_Segmentation.bat"


num_cpus  <- detectCores()
used_cpus <- max(1, num_cpus - 5)
cat("Using", used_cpus, "CPUs for processing...\n")


process_image <- function(image_path, otb_exe, out_dir, spat_r, ran_r) {
  
  file_name <- tools::file_path_sans_ext(basename(image_path))
  output_shp <- file.path(out_dir, paste0(file_name, "_seg.shp"))
  
  
  command <- paste(
    otb_exe,
    "-in", image_path,
    "-filter meanshift",
    "-filter.meanshift.spatialr", spat_r,
    "-filter.meanshift.ranger", ran_r,
    "-mode vector",
    "-mode.vector.out", output_shp,
    sep = " "
  )
  
  system(command)
}


start_time <- Sys.time()

if (.Platform$OS.type == "unix") {
  mclapply(tif_files, function(f) {
    process_image(f, otb_exe = otb_path, out_dir = output_base, spat_r = spatial_radius, ran_r = range_radius)
  }, mc.cores = used_cpus)
} else {
  cl <- makeCluster(used_cpus)
  
  clusterExport(cl, c("process_image", "otb_path", "output_base", "spatial_radius", "range_radius"))
  parLapply(cl, tif_files, function(f) {
    process_image(f, otb_exe = otb_path, out_dir = output_base, spat_r = spatial_radius, ran_r = range_radius)
  })
  stopCluster(cl)
}

end_time <- Sys.time()
cat("Finished! Total time:", end_time - start_time, "\n")