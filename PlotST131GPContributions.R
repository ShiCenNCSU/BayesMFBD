#!/usr/bin/env Rscript
# Ternary posterior display of relative fitted-variation contributions.
# Supports linear, B-spline and GP fits; the filename is retained for compatibility.
# Run: Rscript PlotST131GPContributions.R [ST131_GP_contributions_directory] [output_directory]
# Each posterior draw is closed separately: share_k = C_k / sum(C_main, C_interaction, C_nonlinear).
# All posterior draws are shown. The 95% region follows their support dimension.

suppressPackageStartupMessages(library(ggplot2))
required <- c("MASS", "jsonlite", "svglite", "ragg")
for (package in required) {
  if (!requireNamespace(package, quietly = TRUE)) stop("Required R package unavailable: ", package)
}
args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 2) stop("Usage: Rscript PlotST131GPContributions.R [input_directory] [output_directory]")
input_dir <- if (length(args)) args[1] else "ST131_GP_contributions"
out_dir <- if (length(args) > 1) args[2] else file.path(input_dir, "figures")
draw_file <- file.path(input_dir, "component_draws.csv")
if (!file.exists(draw_file)) stop("Missing component_draws.csv; rerun ST131GPContributions.py to export paired draws")
draws <- read.csv(draw_file, stringsAsFactors = FALSE)
fit_metadata <- jsonlite::fromJSON(file.path(input_dir, "analysis_metadata.json"))
fitted_model <- fit_metadata$fitted_configuration$model
feature_set <- fit_metadata$fitted_configuration$feature_set
family_labels <- c(linear = "linear", bspline_unconstrained = "unconstrained B-spline",
                   bspline_monotone = "monotone B-spline", bspline_unimodal = "unimodal B-spline",
                   gp_monotone = "monotone GP")
if (is.null(fitted_model) || !fitted_model %in% names(family_labels)) stop("Missing or unknown fitted model in metadata")
model_title <- paste("ST131:", if (feature_set == "full") "Full" else "Reduced", family_labels[[fitted_model]])
filtered <- identical(fit_metadata$interaction_selection$mode, "significant")
unassigned <- identical(fit_metadata$excluded_interaction_policy, "unassigned")
selection_note <- if (filtered) paste0(fit_metadata$interaction_selection$n_selected, "/",
                                      fit_metadata$interaction_selection$n_total, " interactions retained; ",
                                      100 * fit_metadata$interaction_selection$credible_mass,
                                      "% coefficient CrI excludes zero.\n",
                                      if (unassigned) "Excluded interactions are unassigned; original nonlinear component retained."
                                      else "Nonlinear component includes excluded interactions.") else NULL
coverage <- if (unassigned) read.csv(file.path(input_dir, "coverage_summary.csv")) else NULL
coverage_note <- if (unassigned) sprintf("Unassigned fitted variation: %.1f%% (mean); outside the ternary shares.",
                                         100 * coverage$posterior_mean[coverage$metric == "unassigned_fraction"]) else NULL
divergences <- fit_metadata$sampler_diagnostics$divergences
diagnostic_note <- paste0(fit_metadata$n_chains, " chain(s)",
                          if (!is.null(divergences)) paste0("; ", divergences, " sampler divergences") else "",
                          ". Check convergence before interpretation.")
components <- c("main", "interaction", "nonlinear")
raw_columns <- paste0("C_", components)
if (!all(c("chain", "draw", raw_columns) %in% names(draws))) stop("Missing contribution or chain/draw columns")
if (anyDuplicated(draws[c("chain", "draw")])) stop("Posterior chain/draw pairs must be unique")
raw <- as.matrix(draws[raw_columns])
if (nrow(raw) < 3 || any(!is.finite(raw))) stop("Ternary plotting requires at least three fully defined contribution draws")
if (any(raw < -1e-12)) stop("Unique contributions must be nonnegative for ternary normalization")
roundoff_corrections <- sum(raw < 0)
raw[raw < 0] <- 0 # Only negative roundoff at most 1e-12 is corrected and counted.
totals <- rowSums(raw)
if (any(totals <= 0)) stop("The contribution sum must be positive in every draw")
shares <- raw / totals
if (any(abs(rowSums(shares) - 1) > 1e-12)) stop("Normalized shares must sum to one in each draw")
height <- sqrt(3) / 2
draws$share_main <- shares[, 1]
draws$share_interaction <- shares[, 2]
draws$share_nonlinear <- shares[, 3]
draws$x <- shares[, 2] + 0.5 * shares[, 3]
draws$y <- height * shares[, 3]
# Check the inverse barycentric transform so axis identities cannot be swapped.
recovered_nonlinear <- draws$y / height
recovered_interaction <- draws$x - 0.5 * recovered_nonlinear
stopifnot(max(abs(recovered_interaction - shares[, 2])) < 1e-12,
          max(abs(1 - recovered_interaction - recovered_nonlinear - shares[, 1])) < 1e-12)
summary <- data.frame(component = components, posterior_mean = colMeans(shares),
                      posterior_median = apply(shares, 2, median),
                      ci_lower = apply(shares, 2, quantile, probs = 0.025),
                      ci_upper = apply(shares, 2, quantile, probs = 0.975),
                      credible_mass = 0.95, n_draws = nrow(draws), row.names = NULL)
center <- data.frame(x = summary$posterior_mean[2] + 0.5 * summary$posterior_mean[3],
                     y = height * summary$posterior_mean[3])

# A density-based region preserves posterior asymmetry when support is 2D. The threshold is the
# 5th percentile of KDE density evaluated at the actual posterior draws,
# giving 95% empirical posterior coverage, up to grid interpolation/ties.
# It is an approximate region, rather than a Gaussian confidence ellipse.
support <- eigen(cov(cbind(draws$x, draws$y)), symmetric = TRUE)
support_rank <- sum(support$values > max(1e-14, support$values[1] * 1e-10))
density_threshold <- NULL
interval_segment <- NULL
if (support_rank == 2) {
kde <- MASS::kde2d(draws$x, draws$y, n = c(350, 350), lims = c(0, 1, 0, height))
grid_density_at_draws <- function(x, y, kde) {
  ix <- findInterval(x, kde$x, all.inside = TRUE)
  iy <- findInterval(y, kde$y, all.inside = TRUE)
  tx <- (x - kde$x[ix]) / (kde$x[ix + 1] - kde$x[ix])
  ty <- (y - kde$y[iy]) / (kde$y[iy + 1] - kde$y[iy])
  (1 - tx) * (1 - ty) * kde$z[cbind(ix, iy)] +
    tx * (1 - ty) * kde$z[cbind(ix + 1, iy)] +
    (1 - tx) * ty * kde$z[cbind(ix, iy + 1)] +
    tx * ty * kde$z[cbind(ix + 1, iy + 1)]
}
draws$posterior_density <- grid_density_at_draws(draws$x, draws$y, kde)
density_threshold <- unname(quantile(draws$posterior_density, 0.05))
draws$inside_approx_95_region <- draws$posterior_density >= density_threshold
density_grid <- expand.grid(x = kde$x, y = kde$y)
density_grid$density <- as.vector(kde$z)
inside_simplex <- density_grid$y >= 0 & density_grid$y <= height &
  density_grid$x >= density_grid$y / (2 * height) &
  density_grid$x <= 1 - density_grid$y / (2 * height)
# Mask only the KDE evaluation grid outside the simplex; no draws are removed.
density_grid$density[!inside_simplex] <- NA_real_
region_note <- "Outline/shading: approximate 95% density region (KDE)."
region_method <- "approximate KDE density region; density threshold is 5th percentile over paired posterior draws"
} else if (support_rank == 1) {
  # Zero interaction contribution puts every draw on one simplex edge.
  # Preserve this support instead of smoothing a spurious 2D density around it.
  origin <- colMeans(cbind(draws$x, draws$y))
  direction <- support$vectors[, 1]
  projection <- sweep(cbind(draws$x, draws$y), 2, origin) %*% direction
  interval <- unname(quantile(projection, c(.025, .975)))
  endpoints <- rbind(origin + interval[1] * direction, origin + interval[2] * direction)
  interval_segment <- data.frame(x = endpoints[1, 1], y = endpoints[1, 2],
                                 xend = endpoints[2, 1], yend = endpoints[2, 2])
  draws$posterior_density <- NA_real_
  draws$inside_approx_95_region <- projection[, 1] >= interval[1] & projection[, 1] <= interval[2]
  region_note <- "Thick segment: central 95% interval along the one-dimensional support."
  region_method <- "equal-tailed 95 percent interval of projection onto one-dimensional support"
} else {
  draws$posterior_density <- NA_real_
  draws$inside_approx_95_region <- TRUE
  region_note <- "All draws have identical relative shares; no density region is drawn."
  region_method <- "point support; no density region"
}

# Equilateral barycentric grid: main effects read along the bottom edge,
# interactions along the right edge, and nonlinearity along the left edge.
grid_lines <- do.call(rbind, lapply(seq(0.2, 0.8, by = 0.2), function(q) {
  data.frame(x = c(1 - q, q, 0.5 * q),
             y = c(0, 0, height * q),
             xend = c(0.5 * (1 - q), q + 0.5 * (1 - q), 1 - 0.5 * q),
             yend = c(height * (1 - q), height * (1 - q), height * q))
}))
triangle <- data.frame(x = c(0, 1, 0.5, 0), y = c(0, 0, height, 0))
q <- seq(0.2, 0.8, by = 0.2)
tick_labels <- rbind(data.frame(x = 1 - q, y = -0.026, label = as.character(round(100 * q))),
                     data.frame(x = 0.5 * q - 0.037, y = height * q + 0.01,
                                label = as.character(round(100 * q))),
                     data.frame(x = q + 0.5 * (1 - q) + 0.037, y = height * (1 - q) + 0.01,
                                label = as.character(round(100 * q))))
vertex_labels <- data.frame(x = c(0, 1, 0.5), y = c(-0.094, -0.094, height + 0.076),
                            name = c("Main effects", if (filtered) "Significant interactions" else "Interactions",
                                     "Nonlinear"))
vertex_labels$estimate <- sprintf("%.1f%% [%.1f, %.1f]", 100 * summary$posterior_mean,
                                  100 * summary$ci_lower, 100 * summary$ci_upper)
vertex_labels$estimate_y <- vertex_labels$y - 0.032
caption <- paste0("Per draw: relative share = unique contribution / sum of all three contributions.\n",
                  "Dots: all ", format(nrow(draws), big.mark = ","), " posterior draws; diamond: mean normalized shares.\n",
                  region_note, "\nVertex values: mean [marginal 95% CrI].\n",
                  if (filtered) paste0(selection_note, "\n") else "",
                  if (unassigned) paste0(coverage_note, "\n") else "", diagnostic_note)

theme_set(theme_void(base_size = 8, base_family = "Helvetica") +
            theme(plot.title = element_text(size = 11, face = "bold", hjust = 0),
                  plot.subtitle = element_text(size = 8, colour = "#555555", hjust = 0, margin = margin(t = 4)),
                  plot.caption = element_text(size = 7, colour = "#444444", hjust = 0, lineheight = 1.15,
                                              margin = margin(t = 10)),
                  plot.margin = margin(12, 12, 10, 12, unit = "pt")))
plot <- ggplot() +
  geom_segment(data = grid_lines, aes(x, y, xend = xend, yend = yend),
               linewidth = 0.3, colour = "#DDDDDD")
if (support_rank == 2) plot <- plot +
  geom_contour_filled(data = density_grid, aes(x, y, z = density),
                      breaks = c(density_threshold, max(kde$z) + 1),
                      fill = "#94C3CF", alpha = 0.25, show.legend = FALSE, na.rm = TRUE)
point_layer <- length(plot$layers) + 1
plot <- plot + geom_point(data = draws, aes(x, y), colour = "#185C7A", alpha = 0.22, size = 0.65)
if (support_rank == 2) plot <- plot +
  geom_contour(data = density_grid, aes(x, y, z = density), breaks = density_threshold,
               colour = "#185C7A", linewidth = 0.6, na.rm = TRUE)
if (support_rank == 1) plot <- plot +
  geom_segment(data = interval_segment, aes(x, y, xend = xend, yend = yend),
               colour = "#185C7A", linewidth = 1.8)
plot <- plot +
  geom_path(data = triangle, aes(x, y), colour = "#555555", linewidth = 0.55) +
  geom_point(data = center, aes(x, y), shape = 23, size = 3.2, stroke = 0.65,
             fill = "#E3A34B", colour = "#292929") +
  geom_text(data = tick_labels, aes(x, y, label = label), family = "Helvetica", size = 7 / ggplot2::.pt,
            colour = "#555555") +
  geom_text(data = vertex_labels, aes(x, y, label = name), family = "Helvetica", size = 9 / ggplot2::.pt,
            fontface = "bold", colour = "#333333") +
  geom_text(data = vertex_labels, aes(x, estimate_y, label = estimate), family = "Helvetica",
            size = 7.5 / ggplot2::.pt, colour = "#444444") +
  coord_fixed(ratio = 1, xlim = c(-0.19, 1.19), ylim = c(-0.16, height + 0.14), expand = FALSE, clip = "off") +
  labs(title = model_title,
       subtitle = "Relative component contributions and posterior uncertainty", caption = caption)

# Assert the scatter preserves every draw and its paired ternary coordinates.
built <- ggplot_build(plot)
marks <- built$data[[point_layer]]
stopifnot(nrow(marks) == nrow(draws), max(abs(marks$x - draws$x)) < 1e-12,
          max(abs(marks$y - draws$y)) < 1e-12)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
stem <- file.path(out_dir, if (fitted_model == "gp_monotone") "gp_component_contributions_ternary" else
                   paste0(fit_metadata$candidate, "_component_contributions_ternary"))
width_mm <- 183
height_mm <- 165
# Native PDF retains editable Helvetica text without requiring XQuartz/Cairo.
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
write.csv(draws, file.path(out_dir, "normalized_draws.csv"), row.names = FALSE)
write.csv(summary, file.path(out_dir, "normalized_summary.csv"), row.names = FALSE)
metadata <- list(candidate = fit_metadata$candidate, model = fitted_model, title = model_title,
                 normalization = "share_component_draw = C_component_draw / sum(C_main_draw, C_interaction_draw, C_nonlinear_draw)",
                 center = "posterior_mean_of_draw_normalized_shares", marginal_interval = "equal_tailed_95_percent",
                 region = region_method, support_dimension = support_rank,
                 kde_grid_size = if (support_rank == 2) c(350, 350) else NULL, density_threshold = density_threshold,
                 empirical_region_coverage = mean(draws$inside_approx_95_region),
                 simplex_mask = "only KDE grid cells outside the barycentric triangle are masked",
                 n_input_draws = nrow(draws), n_plotted_draws = nrow(marks), n_excluded_draws = 0,
                 n_negative_roundoff_values_corrected = roundoff_corrections,
                 n_chains = fit_metadata$n_chains, draws_per_chain = fit_metadata$draws_per_chain,
                 contribution_row_weighting = fit_metadata$row_weighting, contribution_rows = fit_metadata$rows,
                 posterior_sha256 = fit_metadata$posterior_sha256, branch_effects_included = FALSE,
                 interaction_selection = fit_metadata$interaction_selection,
                 nonlinear_includes_excluded_interactions = filtered && !unassigned,
                 excluded_interactions_unassigned = unassigned,
                 retained_regression_coverage = coverage,
                 contribution_definition = fit_metadata$contribution_definition,
                 width_mm = width_mm, height_mm = height_mm, png_dpi = 300,
                 backend = "R/ggplot2", R_version = R.version.string,
                 ggplot2_version = as.character(packageVersion("ggplot2")),
                 figure_contract = list(question = "What is the joint posterior distribution of relative component contributions?",
                                        claim = paste("Posterior uncertainty in relative component contributions for", fit_metadata$candidate),
                                        archetype = "quantitative grid; one ternary panel", role = "descriptive model attribution",
                                        alignment_audit = "not applicable: one plot panel"))
jsonlite::write_json(metadata, paste0(stem, ".metadata.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
writeLines(c("All posterior draws are retained and normalized separately; original contributions are unchanged.",
             "The ternary figure describes relative unique contributions, not an additive partition of total fitted variance.",
             region_note,
             "Marginal intervals are not simultaneous joint intervals.",
             if (filtered) selection_note else "All fitted interaction terms enter interaction attribution.",
             if (unassigned) coverage_note else "",
             diagnostic_note,
             "Grid lines and percentage ticks map main effects to the left vertex, interactions to the right, and the nonlinear component to the top.",
             "Native PDF uses editable Helvetica text; SVG and 300-dpi PNG are also exported."),
           file.path(out_dir, "figure_notes.txt"))
message("Saved ", stem, ".pdf/.svg/.png; all ", nrow(draws), " draws shown")
