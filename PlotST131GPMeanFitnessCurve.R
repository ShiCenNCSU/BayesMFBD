#!/usr/bin/env Rscript
# Run: Rscript PlotST131GPMeanFitnessCurve.R GP_SUMMARY_DIRECTORY [x_min x_max]
# Log-scale GP fits need draw-based curve summaries. Legacy fitness-scale
# GP fits can obtain their exact mean curve from posterior mean knots.
suppressPackageStartupMessages(library(ggplot2))
for (package in c("jsonlite", "svglite", "ragg")) {
  if (!requireNamespace(package, quietly = TRUE)) stop("Missing R package: ", package)
}
args <- commandArgs(trailingOnly = TRUE)
if (!length(args) %in% c(1, 3)) stop("Usage: Rscript PlotST131GPMeanFitnessCurve.R SUMMARY_DIRECTORY [x_min x_max]")
input_dir <- args[1]
x_limits <- if (length(args) == 3) suppressWarnings(as.numeric(args[2:3])) else c(-3, 3)
if (any(!is.finite(x_limits)) || x_limits[1] >= x_limits[2]) stop("Invalid latent-score range")
config <- jsonlite::fromJSON(file.path(input_dir, "run_config.json"))
metadata <- jsonlite::fromJSON(file.path(input_dir, "contributions", "analysis_metadata.json"))
if (config$model != "gp_monotone" || metadata$model != "gp_monotone") stop("A fitted monotone GP is required")
log_scale_gp <- identical(config$gp_output_scale, "log")
if (log_scale_gp) {
  curve_source <- file.path("fitness_curve", "curve_summary.csv")
  curve_path <- file.path(input_dir, curve_source)
  if (!file.exists(curve_path)) {
    stop("Log-scale GP mean curves require fitness_curve/curve_summary.csv from paired posterior draws; run ST131GPFitnessCurve.py or SummarizeST131.py first. Knot means alone are insufficient.")
  }
  curve <- read.csv(curve_path, check.names = FALSE)
  columns <- c("latent_score", "cdf_coordinate", "fitness_mean")
  if (!all(columns %in% names(curve))) stop("Draw-based curve summary is missing required columns")
  curve <- curve[, columns, drop = FALSE]
  if (nrow(curve) < 2L || any(!is.finite(as.matrix(curve))) ||
      any(diff(curve$latent_score) <= 0) || any(curve$fitness_mean <= 0)) stop("Invalid draw-based fitness curve")
  curve <- curve[curve$latent_score >= x_limits[1] & curve$latent_score <= x_limits[2], , drop = FALSE]
  if (nrow(curve) < 2L) stop("The requested score range must contain at least two exported curve coordinates")
  curve_estimator <- "posterior mean of exponentiated interpolated log fitness from all paired posterior draws"
  mapping_note <- "For every posterior draw, the inference mapping interpolates log-fitness knots at the normal CDF of the score and then exponentiates. This preview uses the exported arithmetic mean of those draw-wise fitness predictions; interpolating knot means would give a different curve."
} else {
  if (!is.null(config$gp_output_scale) && config$gp_output_scale != "fitness") stop("Unknown GP output scale")
  curve_source <- "HMC_summary_fitness_knots.csv"
  knots <- read.csv(file.path(input_dir, curve_source), row.names = 1, check.names = FALSE)
  expected_names <- paste0("fitness_knots_", seq_len(config$gp_knots) - 1L)
  if (!setequal(rownames(knots), expected_names)) stop("Saved knot summaries differ from the fitted grid")
  knots <- knots[expected_names, , drop = FALSE]
  if (any(!is.finite(knots$mean)) || any(knots$mean <= 0) || any(diff(knots$mean) < 0)) stop("Invalid mean fitness knots")
  curve <- data.frame(latent_score = seq(x_limits[1], x_limits[2], length.out = 501L))
  curve$cdf_coordinate <- pnorm(curve$latent_score)
  curve$fitness_mean <- approx(seq(0, 1, length.out = config$gp_knots), knots$mean,
                               xout = curve$cdf_coordinate, method = "linear", rule = 2)$y
  curve_estimator <- "exact posterior mean fitness from linear interpolation of posterior mean fitness knots"
  mapping_note <- "The inference mapping interpolates fitness knots linearly at the normal CDF of each fixed score. Linearity means interpolating their posterior means reproduces the posterior mean response exactly. No additional smoother is applied."
}
stopifnot(all(is.finite(curve$fitness_mean)), all(diff(curve$fitness_mean) >= -1e-12))
plot <- ggplot(curve, aes(latent_score, fitness_mean, colour = "GP posterior mean")) +
  geom_line(linewidth = 0.65) +
  scale_colour_manual(values = c("GP posterior mean" = "#0072B2"), name = NULL) +
  scale_x_continuous(limits = x_limits, expand = expansion(mult = 0.015)) +
  scale_y_continuous(expand = expansion(mult = 0.035)) +
  labs(x = "Latent additive score", y = "Fitness") +
  theme_classic(base_size = 8, base_family = "Helvetica") +
  theme(axis.text = element_text(size = 7, colour = "#333333"),
        axis.title = element_text(size = 8),
        axis.line = element_line(linewidth = 0.3, colour = "#444444"),
        axis.ticks = element_line(linewidth = 0.25, colour = "#444444"),
        legend.position = "bottom", legend.text = element_text(size = 7),
        legend.key.width = grid::unit(16, "pt"),
        plot.margin = margin(6, 8, 6, 6, unit = "pt"))
built <- ggplot_build(plot)
stopifnot(nrow(built$data[[1]]) == nrow(curve),
          max(abs(built$data[[1]]$y - curve$fitness_mean)) < 1e-12)
out_dir <- file.path(input_dir, "fitness_curve_mean_preview", "figures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
stem <- file.path(out_dir, paste0(metadata$candidate, "_fitness_vs_latent_mean_preview"))
width_mm <- 183
height_mm <- 105
grDevices::pdf(paste0(stem, ".pdf"), width = width_mm / 25.4, height = height_mm / 25.4,
               family = "Helvetica", useDingbats = FALSE)
print(plot)
grDevices::dev.off()
svglite::svglite(paste0(stem, ".svg"), width = width_mm / 25.4, height = height_mm / 25.4)
print(plot)
grDevices::dev.off()
ragg::agg_png(paste0(stem, ".png"), width = width_mm, height = height_mm, units = "mm", res = 300)
print(plot)
grDevices::dev.off()
write.csv(curve, paste0(stem, ".source_data.csv"), row.names = FALSE)
report <- list(candidate = metadata$candidate, model = config$model, feature_set = config$feature_set,
               n_total_draws = metadata$n_total_draws, n_chains = metadata$n_chains,
               draws_per_chain = metadata$draws_per_chain, posterior_sha256 = metadata$posterior_sha256,
               curve_estimator = curve_estimator,
               uncertainty_available = FALSE, genotype_fitness_points_available = FALSE,
               source = curve_source, n_curve_coordinates = nrow(curve),
               x_display_range = x_limits, width_mm = width_mm, height_mm = height_mm,
               sampler_diagnostics = metadata$sampler_diagnostics,
               panel_alignment = "not applicable: single panel")
jsonlite::write_json(report, paste0(stem, ".metadata.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
writeLines(c("# GP fitness response: posterior mean preview", "",
             sprintf("The blue line is the posterior mean genetic fitness at fixed latent additive scores, using all %s draws across %d chains. The score includes the fitted intercept and normalized main and pairwise effects; branch random effects are excluded.",
                     format(metadata$n_total_draws, big.mark = ","), metadata$n_chains), "",
             mapping_note, "",
             if (log_scale_gp) "This mean-only preview omits credible bands and genotype points. The full fitness-curve export contains those draw-based summaries." else "Paired posterior draws are needed to calculate credible bands and genotype-specific fitness summaries. Those are pending the cluster fitness-curve export and are omitted from this mean-only preview.", "",
             sprintf("The displayed score grid is [%g, %g]. The fit reports %d divergences.", x_limits[1], x_limits[2], metadata$sampler_diagnostics$divergences)),
           paste0(stem, ".legend.md"))
message("Saved exact GP mean preview: ", stem, ".pdf/.svg/.png")
