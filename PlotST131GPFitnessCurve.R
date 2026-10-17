#!/usr/bin/env Rscript
# Run: Rscript PlotST131GPFitnessCurve.R [summary_directory] [linear or log] [x_min x_max]
# R draws and exports the fitted response; no refitting or smoothing.
suppressPackageStartupMessages(library(ggplot2))
for (package in c("jsonlite", "svglite", "ragg")) {
  if (!requireNamespace(package, quietly = TRUE)) stop("Missing R package: ", package)
}
args <- commandArgs(trailingOnly = TRUE)
if (!length(args) %in% c(0, 1, 2, 4)) stop("Usage: Rscript PlotST131GPFitnessCurve.R [input_directory] [linear or log] [x_min x_max]")
input_dir <- if (length(args)) args[1] else "ST131_GP_fitness_curve"
y_scale <- if (length(args) >= 2) args[2] else "linear"
if (!y_scale %in% c("linear", "log")) stop("Y scale must be linear or log")
display_y <- if (y_scale == "log") log10 else identity
x_view <- if (length(args) == 4) suppressWarnings(as.numeric(args[3:4])) else NULL
if (!is.null(x_view) && (any(!is.finite(x_view)) || x_view[1] >= x_view[2])) {
  stop("x_min and x_max must be finite and increasing")
}
curve <- read.csv(file.path(input_dir, "curve_summary.csv"))
tips <- read.csv(file.path(input_dir, "tip_summary.csv"),
                 colClasses = c(tip = "character", genotype = "character"))
genotypes <- read.csv(file.path(input_dir, "genotype_summary.csv"),
                      colClasses = c(genotype = "character"))
metadata <- jsonlite::fromJSON(file.path(input_dir, "analysis_metadata.json"))
# Select saved marginal medians without changing the source summaries or intervals.
if (!all(c("fitness_median", "fitness_lower", "fitness_upper") %in% names(curve)) ||
    !all(c("latent_median", "fitness_median") %in% names(genotypes)) ||
    !"latent_median" %in% names(tips)) stop("Posterior-median summaries are required")
stopifnot(all(curve$fitness_lower <= curve$fitness_median),
          all(curve$fitness_median <= curve$fitness_upper),
          all(genotypes$latent_lower <= genotypes$latent_median),
          all(genotypes$latent_median <= genotypes$latent_upper),
          all(genotypes$fitness_lower <= genotypes$fitness_median),
          all(genotypes$fitness_median <= genotypes$fitness_upper))
model_labels <- c(gp_monotone = "GP", bspline_unconstrained = "B-spline",
                  bspline_monotone = "B-spline", bspline_unimodal = "B-spline", linear = "Linear")
model_descriptions <- c(gp_monotone = "monotone GP", bspline_unconstrained = "unconstrained B-spline",
                        bspline_monotone = "monotone B-spline", bspline_unimodal = "unimodal B-spline",
                        linear = "linear log-fitness")
if (!metadata$model %in% names(model_labels)) stop("Unknown fitted model family")
model_label <- unname(model_labels[metadata$model])
model_description <- unname(model_descriptions[metadata$model])
median_label <- paste(model_label, "median")
if (nrow(curve) != metadata$n_grid_points || nrow(tips) != metadata$n_sampled_tips) stop("Summary counts differ from metadata")
if (any(!is.finite(as.matrix(curve))) || any(diff(curve$latent_score) <= 0)) stop("The curve grid must be finite and increasing")
if (any(curve$fitness_lower <= 0) || any(curve$fitness_lower > curve$fitness_upper) ||
    any(curve$fitness_median <= 0)) stop("Invalid fitness estimates or credible intervals")
if (any(!is.finite(tips$latent_median)) || any(tips$latent_median < min(curve$latent_score)) ||
    any(tips$latent_median > max(curve$latent_score))) stop("A sampled tip lies outside the plotted latent range")
if (metadata$model %in% c("gp_monotone", "bspline_monotone", "linear") &&
    any(diff(curve$fitness_median) < -1e-12)) stop("The fitted monotone response is not monotone")
if (nrow(genotypes) != metadata$n_unique_genotypes || anyDuplicated(genotypes$genotype) ||
    sum(genotypes$sampled_tips) != nrow(tips)) stop("Genotype points do not represent every sampled tip")
point_columns <- c("latent_lower", "latent_median", "latent_upper",
                   "fitness_lower", "fitness_median", "fitness_upper")
# Medians must lie within their equal-tailed intervals; means remain in the source tables.
if (any(!is.finite(as.matrix(genotypes[point_columns]))) ||
    any(genotypes$latent_lower > genotypes$latent_upper) ||
    any(genotypes$fitness_lower <= 0) ||
    any(genotypes$fitness_median <= 0) ||
    any(genotypes$fitness_lower > genotypes$fitness_upper)) stop("Invalid genotype point estimates or intervals")
if (min(genotypes$latent_lower) < min(curve$latent_score) ||
    max(genotypes$latent_upper) > max(curve$latent_score)) stop("A latent-score interval would be clipped")
tip_counts <- table(tips$genotype)
if (!setequal(names(tip_counts), genotypes$genotype) ||
    !all(as.integer(tip_counts[genotypes$genotype]) == genotypes$sampled_tips)) stop("Genotype carrier counts differ from the tip table")
interval_label <- sprintf("%g%% %s credible band", metadata$credible_mass * 100, model_label)
curve_display <- curve[c("latent_score", "fitness_median", "fitness_lower", "fitness_upper")]
if (!is.null(x_view)) {
  boundary_scores <- x_view[x_view >= min(curve$latent_score) & x_view <= max(curve$latent_score) &
                           !x_view %in% curve$latent_score]
  boundary <- data.frame(latent_score = boundary_scores)
  for (column in c("fitness_median", "fitness_lower", "fitness_upper")) {
    values <- approx(curve$latent_score, display_y(curve[[column]]), xout = boundary_scores)$y
    boundary[[column]] <- if (y_scale == "log") 10^values else values
  }
  # Keep the existing rendered interpolation at the requested crop boundaries.
  curve_display <- rbind(curve_display[curve_display$latent_score >= x_view[1] &
                                       curve_display$latent_score <= x_view[2], ], boundary)
  curve_display <- curve_display[order(curve_display$latent_score), ]
}
# Retain full intervals in source tables while restricting rendered paths to
# the viewport, so hidden paths cannot cross PDF tick-label geometry.
horizontal_genotypes <- genotypes
if (!is.null(x_view)) {
  horizontal_genotypes <- horizontal_genotypes[horizontal_genotypes$latent_upper >= x_view[1] &
                                               horizontal_genotypes$latent_lower <= x_view[2], ]
  horizontal_genotypes$latent_lower <- pmax(horizontal_genotypes$latent_lower, x_view[1])
  horizontal_genotypes$latent_upper <- pmin(horizontal_genotypes$latent_upper, x_view[2])
}
theme_set(theme_classic(base_size = 8, base_family = "Helvetica") +
            theme(axis.text = element_text(size = 7, colour = "#333333"),
                  axis.title = element_text(size = 8),
                  axis.line = element_line(linewidth = 0.3, colour = "#444444"),
                  axis.ticks = element_line(linewidth = 0.25, colour = "#444444"),
                  legend.position = "bottom", legend.title = element_blank(),
                  legend.text = element_text(size = 7),
                  legend.key.width = grid::unit(16, "pt"),
                  legend.key.height = grid::unit(10, "pt"),
                  legend.box = "horizontal", legend.box.spacing = grid::unit(5, "pt"),
                  plot.title = element_blank(), plot.subtitle = element_blank(),
                  plot.caption = element_blank(),
                  plot.margin = margin(6, 8, 6, 6, unit = "pt")))
plot <- ggplot(curve_display, aes(x = latent_score)) +
  geom_ribbon(aes(ymin = fitness_lower, ymax = fitness_upper, fill = interval_label),
                alpha = 0.2, colour = NA) +
  geom_line(aes(y = fitness_median, colour = median_label), linewidth = 0.65) +
  geom_segment(data = horizontal_genotypes,
                 aes(x = latent_lower, xend = latent_upper, y = fitness_median, yend = fitness_median),
                 inherit.aes = FALSE, colour = "#D55E00", alpha = 0.22,
                 linewidth = 0.25, show.legend = FALSE) +
  geom_segment(data = genotypes,
                 aes(x = latent_median, xend = latent_median, y = fitness_lower, yend = fitness_upper),
                 inherit.aes = FALSE, colour = "#D55E00", alpha = 0.3,
                 linewidth = 0.25, show.legend = FALSE) +
  geom_point(data = genotypes,
               aes(x = latent_median, y = fitness_median, shape = "Genotype medians"),
               inherit.aes = FALSE, colour = "#D55E00", alpha = 0.85, size = 1.5) +
  geom_rug(data = tips, aes(x = latent_median), inherit.aes = FALSE, sides = "b",
             length = grid::unit(2.5, "pt"), linewidth = 0.25, alpha = 0.2, colour = "#333333") +
  scale_colour_manual(values = setNames("#0072B2", median_label), name = NULL) +
  scale_fill_manual(values = setNames("#0072B2", interval_label), name = NULL) +
  scale_shape_manual(values = c("Genotype medians" = 16), name = NULL) +
  scale_x_continuous(limits = range(curve$latent_score), expand = expansion(mult = 0.015)) +
  scale_y_continuous(transform = if (y_scale == "log") "log10" else "identity",
                     expand = expansion(mult = 0.035)) +
  guides(colour = guide_legend(order = 1), fill = guide_legend(order = 2),
         shape = guide_legend(order = 3)) +
  labs(x = "Latent additive score", y = if (y_scale == "log") "Fitness (log scale)" else "Fitness",
         title = NULL, subtitle = NULL, caption = NULL)
if (!is.null(x_view)) {
  inside <- which(curve$latent_score >= x_view[1] & curve$latent_score <= x_view[2])
  if (!length(inside)) stop("The requested x range does not contain any curve coordinates")
  # Boundary neighbours retain the existing line/ribbon interpolation at the
  # crop edges. Posterior summaries and built data remain unchanged.
  visible <- seq.int(max(1L, min(inside) - 1L), min(nrow(curve), max(inside) + 1L))
  visible_genotypes <- genotypes$latent_median >= x_view[1] & genotypes$latent_median <= x_view[2]
  horizontal_visible <- genotypes$latent_upper >= x_view[1] & genotypes$latent_lower <= x_view[2]
  values <- c(curve$fitness_median[visible], curve$fitness_lower[visible], curve$fitness_upper[visible],
              genotypes$fitness_median[horizontal_visible], genotypes$fitness_lower[visible_genotypes],
              genotypes$fitness_upper[visible_genotypes])
  transformed_limits <- range(display_y(values))
  padding <- max(diff(transformed_limits) * 0.035, 0.01)
  transformed_limits <- transformed_limits + c(-padding, padding)
  y_view <- if (y_scale == "log") 10^transformed_limits else transformed_limits
  plot <- plot + coord_cartesian(xlim = x_view, ylim = y_view, expand = FALSE)
}
built <- ggplot_build(plot)
if (!is.null(x_view)) stopifnot(max(abs(built$layout$panel_params[[1]]$x.range - x_view)) < 1e-12)
stopifnot(nrow(built$data[[1]]) == nrow(curve_display), nrow(built$data[[2]]) == nrow(curve_display),
          nrow(built$data[[3]]) == nrow(horizontal_genotypes), nrow(built$data[[4]]) == nrow(genotypes),
          nrow(built$data[[5]]) == nrow(genotypes), nrow(built$data[[6]]) == nrow(tips),
          max(abs(built$data[[1]]$ymin - display_y(curve_display$fitness_lower))) < 1e-12,
          max(abs(built$data[[1]]$ymax - display_y(curve_display$fitness_upper))) < 1e-12,
          max(abs(built$data[[2]]$y - display_y(curve_display$fitness_median))) < 1e-12,
          max(abs(built$data[[3]]$x - horizontal_genotypes$latent_lower)) < 1e-12,
          max(abs(built$data[[3]]$xend - horizontal_genotypes$latent_upper)) < 1e-12,
          max(abs(built$data[[3]]$y - display_y(horizontal_genotypes$fitness_median))) < 1e-12,
          max(abs(built$data[[4]]$x - genotypes$latent_median)) < 1e-12,
          max(abs(built$data[[4]]$y - display_y(genotypes$fitness_lower))) < 1e-12,
          max(abs(built$data[[4]]$yend - display_y(genotypes$fitness_upper))) < 1e-12,
          max(abs(built$data[[5]]$x - genotypes$latent_median)) < 1e-12,
          max(abs(built$data[[5]]$y - display_y(genotypes$fitness_median))) < 1e-12,
          max(abs(built$data[[6]]$x - tips$latent_median)) < 1e-12,
          all(is.finite(built$data[[6]]$x)))
out_dir <- file.path(input_dir, "figures")
if (!is.null(x_view)) out_dir <- file.path(out_dir, sprintf("latent_%g_to_%g", x_view[1], x_view[2]))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
stem <- file.path(out_dir, paste0(metadata$candidate, "_fitness_vs_latent",
                                if (y_scale == "log") "_log_y" else ""))
width_mm <- 120
height_mm <- 105
# Native PDF preserves editable Helvetica without the local Cairo dependency.
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
ragg::agg_tiff(paste0(stem, ".tiff"), width = width_mm, height = height_mm, units = "mm",
               res = 600, compression = "lzw")
print(plot)
grDevices::dev.off()
write.csv(curve, paste0(stem, ".source_data.csv"), row.names = FALSE)
write.csv(genotypes, paste0(stem, ".genotype_source_data.csv"), row.names = FALSE)
report <- list(candidate = metadata$candidate, backend = "R/ggplot2", n_panels = 1L,
               posterior_summary = "median", posterior_sha256 = metadata$input_hashes$posterior_sha256,
               width_mm = width_mm, height_mm = height_mm, png_dpi = 300L, tiff_dpi = 600L,
               n_curve_coordinates = nrow(curve), n_rendered_curve_coordinates = nrow(curve_display),
               n_rug_marks = nrow(tips),
               n_genotype_points = nrow(genotypes), n_horizontal_intervals = nrow(horizontal_genotypes),
               n_vertical_intervals = nrow(genotypes), n_represented_tips = sum(genotypes$sampled_tips),
               n_excluded_draws = 0L, n_excluded_tips = 0L, credible_mass = metadata$credible_mass,
               n_chains = metadata$n_chains, draws_per_chain = metadata$draws_per_chain,
               model = metadata$model, model_description = model_description,
               sampler_diagnostics = metadata$diagnostics,
               uncertainty = "pointwise equal-tailed fitness credible interval at fixed latent score",
               curve_estimator = "posterior median fitness at each fixed latent score",
               point_estimator = "marginal posterior medians from draw-paired genotype latent scores and fitness",
               point_intervals = "marginal equal-tailed credible intervals: horizontal latent score; vertical genetic fitness",
               point_weighting = "one point per distinct genotype; sampled-tip multiplicities retained in source data and rug",
               figure_title = FALSE, figure_subtitle = FALSE, embedded_caption = FALSE,
               curve_smoothing = FALSE, y_scale = y_scale, colour = "#0072B2",
               x_display_range = if (is.null(x_view)) range(curve$latent_score) else x_view,
               display_crop_requested = !is.null(x_view),
               n_curve_coordinates_in_x_view = if (is.null(x_view)) nrow(curve) else sum(curve$latent_score >= x_view[1] & curve$latent_score <= x_view[2]),
               n_genotype_medians_outside_x_view = if (is.null(x_view)) 0L else sum(genotypes$latent_median < x_view[1] | genotypes$latent_median > x_view[2]),
               n_rug_marks_outside_x_view = if (is.null(x_view)) 0L else sum(tips$latent_median < x_view[1] | tips$latent_median > x_view[2]),
               n_horizontal_intervals_clipped_by_x_view = if (is.null(x_view)) 0L else sum(genotypes$latent_lower < x_view[1] | genotypes$latent_upper > x_view[2]),
               genotype_colour = "#D55E00",
               panel_alignment = "not applicable: single-panel figure",
               figure_contract = list(question = paste("How does genetic fitness depend on the latent additive layer in the fitted ST131", model_description, "model?"),
                                      claim = "The fitted response and its posterior uncertainty vary along the latent additive score",
                                      archetype = "single quantitative response curve"))
jsonlite::write_json(report, paste0(stem, ".metadata.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
score_definition <- paste0("The latent additive score is the fitted intercept plus the ",
                           if (metadata$model == "linear") "" else "normalized ",
                           "main-feature", if (metadata$feature_set == "full") " and pairwise-interaction" else "",
                           " contributions.")
mapping_definition <- if (metadata$model == "gp_monotone") {
  if (identical(metadata$gp_output_scale, "log")) {
    "Fitness is evaluated using the exact inference mapping: the standard normal CDF transforms the latent score onto the GP grid; log-fitness knots are interpolated and then exponentiated. No additional smoother is applied."
  } else {
    "Fitness is evaluated using the exact inference mapping: the standard normal CDF transforms the latent score onto the GP grid, and the fitted fitness knots are interpolated on the fitness scale. No additional smoother is applied."
  }
} else if (grepl("^bspline", metadata$model)) {
  "Fitness is evaluated using the exact inference mapping: the standard normal CDF transforms the latent score to the fitted cubic B-spline basis; the basis-weighted coefficients give log fitness, which is then exponentiated. No additional smoother or shape constraint is applied during postprocessing."
} else "Fitness is the exponential of the fitted latent log-fitness score. No additional smoother is applied."
legend <- c("# ST131 genetic fitness as a function of the latent additive layer", "",
            sprintf("The blue line shows posterior median genetic fitness under the fitted %s %s model; the shaded band shows the pointwise equal-tailed %g%% credible interval at each fixed latent score. All %s retained draws across %d chain(s) are included.",
                    metadata$feature_set, model_description, 100 * metadata$credible_mass,
                    format(metadata$n_total_draws, big.mark = ","), metadata$n_chains), "",
            score_definition,
            mapping_definition, "",
            sprintf("Orange points show posterior median latent score and posterior median genetic fitness for each of the %d distinct sampled genotypes, representing all %d sampled tips. Horizontal bars show the full equal-tailed %g%% credible interval for latent score; vertical bars show the same interval for genotype-specific fitness. Both summaries use paired latent-score and fitness draws. Identical genotypes share a single point; their sampled-tip counts are retained in the source data.",
                    nrow(genotypes), sum(genotypes$sampled_tips), 100 * metadata$credible_mass), "",
            "The horizontal and vertical intervals are marginal intervals, not a joint credible region. Points need not fall exactly on the median response curve because genotype-specific scores and the fitted mapping vary together across posterior draws.", "",
            sprintf("Rug marks show posterior median latent scores for all %d sampled ST131 tips, representing %d distinct genotypes. The displayed range contains every retained sampled-genotype latent draw, with 5%% padding; the rug identifies the median data-supported region.",
                    metadata$n_sampled_tips, metadata$n_unique_genotypes), "",
            "The blue band describes uncertainty in the response mapping at a fixed latent score; the orange bars separately show genotype-specific uncertainty in both coordinates. Independent branch random effects are excluded. The band is pointwise rather than simultaneous.")
if (metadata$n_chains == 1) legend <- c(legend, "", "The saved posterior contains one chain; convergence assessment remains preliminary.")
if (!is.null(x_view)) {
  legend <- gsub("The displayed range contains every retained sampled-genotype latent draw, with 5% padding;", "The source grid contains every retained sampled-genotype latent draw, with 5% padding;", legend, fixed = TRUE)
  legend <- c(legend, "", sprintf("The displayed latent-score range is restricted to [%g, %g] at the user's request. The vertical axis is scaled to the visible curve and genotype uncertainty. Posterior statistics use all retained draws; the source tables retain the complete grid.", x_view[1], x_view[2]))
}
if (y_scale == "log") legend <- c(legend, "", "Fitness is displayed on a logarithmic axis. Medians and credible intervals are calculated on the original fitness scale; all plotted coordinates and intervals are retained.")
if (!is.null(metadata$diagnostics$divergences) && metadata$diagnostics$divergences > 0) {
  legend <- c(legend, "", sprintf("The saved fit reports %d sampler divergences. These results remain exploratory until the sampling problems are resolved.", metadata$diagnostics$divergences))
}
writeLines(legend, paste0(stem, ".legend.md"), useBytes = TRUE)
message("Saved ", stem, ".pdf/.svg/.png/.tiff; ", nrow(genotypes),
        " genotype points with full latent/fitness intervals; all ", nrow(tips), " sampled tips represented")
