#!/usr/bin/env Rscript
# Standalone reduced B-spline contribution figure, adapted from the GP template.
# Rscript PlotST131BSplineMonotoneReducedContributions.R contributions_directory [output_directory]
suppressPackageStartupMessages(library(ggplot2))
for (package in c("jsonlite", "svglite", "ragg")) {
  if (!requireNamespace(package, quietly = TRUE)) stop("Missing R package: ", package)
}
args <- commandArgs(trailingOnly = TRUE)
if (!length(args) %in% 1:2) stop("Usage: Rscript PlotST131BSplineMonotoneReducedContributions.R contributions_directory [output_directory]")
input_dir <- args[1]
out_dir <- if (length(args) > 1) args[2] else file.path(input_dir, "figures")
candidate <- "bspline_monotone_reduced"
fit_metadata <- jsonlite::fromJSON(file.path(input_dir, "analysis_metadata.json"))
draws <- read.csv(file.path(input_dir, "component_draws.csv"))
components <- c("main", "interaction", "nonlinear")
raw <- as.matrix(draws[paste0("C_", components)])
stopifnot(fit_metadata$candidate == candidate, fit_metadata$model == "bspline_monotone",
          fit_metadata$feature_set == "reduced", nrow(raw) == fit_metadata$n_total_draws,
          !anyDuplicated(draws[c("chain", "draw")]), all(is.finite(raw)), all(raw >= -1e-12),
          all(raw[, 2] == 0))
negative_roundoff_corrections <- sum(raw < 0)
raw[raw < 0] <- 0
stopifnot(all(rowSums(raw) > 0))
shares <- raw / rowSums(raw)
stopifnot(max(abs(rowSums(shares) - 1)) < 1e-12)
height <- sqrt(3) / 2
draws$share_main <- shares[, 1]
draws$share_interaction <- shares[, 2]
draws$share_nonlinear <- shares[, 3]
draws$x <- shares[, 2] + 0.5 * shares[, 3]
draws$y <- height * shares[, 3]
summary <- data.frame(component = components, posterior_mean = colMeans(shares),
                      ci_lower = apply(shares, 2, quantile, probs = 0.025),
                      ci_upper = apply(shares, 2, quantile, probs = 0.975))
center <- data.frame(x = summary$posterior_mean[2] + 0.5 * summary$posterior_mean[3],
                     y = height * summary$posterior_mean[3])
# The posterior has one-dimensional support on the main/nonlinearity edge.
# Preserve this support and show its equal-tailed interval without a 2-D KDE.
edge_interval <- data.frame(x = 0.5 * summary$ci_lower[3], y = height * summary$ci_lower[3],
                            xend = 0.5 * summary$ci_upper[3], yend = height * summary$ci_upper[3])
draws$inside_95_interval <- draws$share_nonlinear >= summary$ci_lower[3] &
  draws$share_nonlinear <= summary$ci_upper[3]
stopifnot(all(draws$share_interaction == 0),
          max(abs(draws$x - 0.5 * draws$y / height)) < 1e-12)

grid_lines <- do.call(rbind, lapply(seq(0.2, 0.8, by = 0.2), function(q) {
  data.frame(x = c(1 - q, q, 0.5 * q), y = c(0, 0, height * q),
             xend = c(0.5 * (1 - q), q + 0.5 * (1 - q), 1 - 0.5 * q),
             yend = c(height * (1 - q), height * (1 - q), height * q))
}))
triangle <- data.frame(x = c(0, 1, 0.5, 0), y = c(0, 0, height, 0))
q <- seq(0.2, 0.8, by = 0.2)
tick_labels <- rbind(data.frame(x = 1 - q, y = -0.035, label = as.character(round(100 * q))),
                     data.frame(x = 0.5 * q - 0.047, y = height * q + 0.01, label = as.character(round(100 * q))),
                     data.frame(x = q + 0.5 * (1 - q) + 0.047, y = height * (1 - q) + 0.01,
                                label = as.character(round(100 * q))))
vertex_labels <- data.frame(x = c(0, 1, 0.5), y = c(-0.112, -0.112, height + 0.1),
                            name = c("Main effects", "Interactions", "Nonlinearity"))
vertex_labels$estimate <- sprintf("%.1f%% [%.1f, %.1f]", 100 * summary$posterior_mean,
                                  100 * summary$ci_lower, 100 * summary$ci_upper)
vertex_labels$estimate_y <- vertex_labels$y - 0.05
ternary_plot <- ggplot() +
  geom_segment(data = grid_lines, aes(x, y, xend = xend, yend = yend), linewidth = 0.3, colour = "#DDDDDD") +
  geom_path(data = triangle, aes(x, y), colour = "#555555", linewidth = 0.45) +
  geom_segment(data = edge_interval, aes(x, y, xend = xend, yend = yend),
                linewidth = 4, colour = "#94C3CF", alpha = 0.6, lineend = "round") +
  geom_point(data = draws, aes(x, y), colour = "#185C7A", alpha = 0.22, size = 0.55) +
  geom_point(data = center, aes(x, y), shape = 23, size = 2.6, stroke = 0.55,
             fill = "#E3A34B", colour = "#292929") +
  geom_text(data = tick_labels, aes(x, y, label = label), family = "Helvetica",
            size = 7 / ggplot2::.pt, colour = "#555555") +
  geom_text(data = vertex_labels, aes(x, y, label = name), family = "Helvetica",
            size = 8 / ggplot2::.pt, fontface = "bold", colour = "#333333") +
  geom_text(data = vertex_labels, aes(x, estimate_y, label = estimate), family = "Helvetica",
            size = 7 / ggplot2::.pt, colour = "#444444") +
  coord_fixed(ratio = 1, xlim = c(-0.22, 1.22), ylim = c(-0.21, height + 0.16), expand = FALSE, clip = "off") +
  theme_void(base_size = 8, base_family = "Helvetica") +
  theme(plot.margin = margin(7, 5, 5, 5, unit = "pt"))


# Verify the rendered posterior points, posterior mean and credible segment.
layers <- ggplot_build(ternary_plot)$data
stopifnot(nrow(layers[[4]]) == nrow(draws),
          max(abs(layers[[4]]$x - draws$x)) < 1e-12,
          max(abs(layers[[4]]$y - draws$y)) < 1e-12,
          max(abs(layers[[5]]$x - center$x)) < 1e-12,
          max(abs(layers[[5]]$y - center$y)) < 1e-12,
          max(abs(layers[[3]]$x - edge_interval$x)) < 1e-12,
          max(abs(layers[[3]]$xend - edge_interval$xend)) < 1e-12)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
stem <- file.path(out_dir, paste0(candidate, "_component_contributions_ternary"))
width_mm <- 120
height_mm <- 110
# One ggplot panel; the multi-panel alignment gate is not applicable.
grDevices::pdf(paste0(stem, ".pdf"), width = width_mm / 25.4, height = height_mm / 25.4,
               family = "Helvetica", useDingbats = FALSE)
print(ternary_plot)
grDevices::dev.off()
svglite::svglite(paste0(stem, ".svg"), width = width_mm / 25.4, height = height_mm / 25.4)
print(ternary_plot)
grDevices::dev.off()
ragg::agg_png(paste0(stem, ".png"), width = width_mm, height = height_mm, units = "mm", res = 300)
print(ternary_plot)
grDevices::dev.off()
ragg::agg_tiff(paste0(stem, ".tiff"), width = width_mm, height = height_mm, units = "mm", res = 600, compression = "lzw")
print(ternary_plot)
grDevices::dev.off()
write.csv(summary, file.path(out_dir, "normalized_summary.csv"), row.names = FALSE)
write.csv(draws, paste0(stem, ".source_data.csv"), row.names = FALSE)
legend <- c("# Relative unique contributions under the reduced monotone B-spline", "",
  sprintf("Dots show all %d posterior draws. Each draw is normalized as C_k/(C_main+C_interaction+C_nonlinear). The diamond marks posterior mean shares; vertex labels report means and marginal equal-tailed 95%% credible intervals.", nrow(draws)), "",
  "Interaction contribution is fixed at zero because this reduced candidate has no latent interaction terms. All posterior draws lie on the main-effect/nonlinearity edge. The blue segment marks the equal-tailed 95% interval along that edge; no two-dimensional KDE region or jitter is applied.", "",
  sprintf("Attribution equally weights %d edge rows and %d birth-event rows. The shares measure relative component importance, not an additive partition of total fitted variance. Independent branch random effects are excluded.", fit_metadata$n_edge_rows, fit_metadata$n_birth_rows), "",
  sprintf("All %d chains and %d draws per chain are included. The saved diagnostics report %d sampler divergences.", fit_metadata$n_chains, fit_metadata$draws_per_chain, fit_metadata$sampler_diagnostics$divergences))
writeLines(legend, paste0(stem, ".legend.md"))
metadata <- list(candidate = candidate, backend = "R/ggplot2", n_panels = 1L,
  width_mm = width_mm, height_mm = height_mm, png_dpi = 300L, tiff_dpi = 600L,
  n_draws = nrow(draws), n_chains = fit_metadata$n_chains, draws_per_chain = fit_metadata$draws_per_chain,
  posterior_sha256 = fit_metadata$posterior_sha256, n_excluded_draws = 0,
  n_contribution_rows = fit_metadata$n_observations, contribution_row_weighting = fit_metadata$row_weighting,
  interval = "equal-tailed 95% interval along the main-effect/nonlinearity edge",
  empirical_interval_coverage = mean(draws$inside_95_interval),
  interaction_contribution = "zero by construction", estimator = "posterior mean of per-draw normalized shares",
  smoothing = FALSE, jitter = FALSE, negative_roundoff_corrections = negative_roundoff_corrections,
  panel_alignment = "not applicable: one ggplot panel",
  figure_contract = list(question = "How is fitted genetic log-fitness variation attributed in the reduced monotone B-spline candidate?",
    claim = "The reduced model assigns the largest mean relative contribution to main effects; interactions are absent by construction.",
    archetype = "single quantitative ternary plot with one-dimensional posterior support"))
jsonlite::write_json(metadata, paste0(stem, ".metadata.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
message("Saved ", stem, ".pdf/.svg/.png/.tiff; all ", nrow(draws), " posterior draws retained")
