#!/usr/bin/env Rscript
# Run from this project: Rscript PlotST131GPCombined.R [project_directory] [output_directory]
# Rebuild both panels from the existing posterior summaries at their final size.
# No inference, smoothing, draw thinning, or changes to credible intervals.
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(patchwork))
for (package in c("MASS", "jsonlite", "svglite", "ragg")) {
  if (!requireNamespace(package, quietly = TRUE)) stop("Missing R package: ", package)
}
args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 2) stop("Usage: Rscript PlotST131GPCombined.R [project_directory] [output_directory]")
project_dir <- if (length(args)) args[1] else "."
out_dir <- if (length(args) > 1) args[2] else file.path(project_dir, "ST131_GP_combined", "figures")
curve_dir <- file.path(project_dir, "ST131_GP_fitness_curve")
contribution_dir <- file.path(project_dir, "ST131_GP_contributions")
curve <- read.csv(file.path(curve_dir, "curve_summary.csv"))
genotypes <- read.csv(file.path(curve_dir, "genotype_summary.csv"), colClasses = c(genotype = "character"))
tips <- read.csv(file.path(curve_dir, "tip_summary.csv"), colClasses = c(tip = "character", genotype = "character"))
curve_metadata <- jsonlite::fromJSON(file.path(curve_dir, "analysis_metadata.json"))
contribution_metadata <- jsonlite::fromJSON(file.path(contribution_dir, "analysis_metadata.json"))
draws <- read.csv(file.path(contribution_dir, "component_draws.csv"))
components <- c("main", "interaction", "nonlinear")
raw <- as.matrix(draws[paste0("C_", components)])
stopifnot(curve_metadata$candidate == "gp_monotone_full",
          contribution_metadata$candidate == curve_metadata$candidate,
          contribution_metadata$posterior_sha256 == curve_metadata$input_hashes$posterior_sha256,
          nrow(curve) == curve_metadata$n_grid_points,
          nrow(genotypes) == curve_metadata$n_unique_genotypes,
          nrow(tips) == curve_metadata$n_sampled_tips,
          sum(genotypes$sampled_tips) == nrow(tips),
          !anyDuplicated(genotypes$genotype), !anyDuplicated(draws[c("chain", "draw")]),
          nrow(raw) == curve_metadata$n_total_draws,
          nrow(raw) == contribution_metadata$n_total_draws,
          all(is.finite(raw)), all(raw >= -1e-12),
          all(is.finite(as.matrix(curve))), all(diff(curve$latent_score) > 0),
          all(diff(curve$fitness_median) >= -1e-12),
          all(curve$fitness_lower > 0), all(curve$fitness_lower <= curve$fitness_upper),
          all(genotypes$latent_lower <= genotypes$latent_upper),
          all(genotypes$fitness_lower <= genotypes$fitness_upper),
          min(genotypes$latent_lower) >= min(curve$latent_score),
          max(genotypes$latent_upper) <= max(curve$latent_score))
stopifnot(all(curve$fitness_lower <= curve$fitness_median),
          all(curve$fitness_median <= curve$fitness_upper),
          all(genotypes$latent_lower <= genotypes$latent_median),
          all(genotypes$latent_median <= genotypes$latent_upper),
          all(genotypes$fitness_lower <= genotypes$fitness_median),
          all(genotypes$fitness_median <= genotypes$fitness_upper))
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
# Confirm that the assembly reproduces the statistics of the standalone figure.
original_summary <- read.csv(file.path(contribution_dir, "figures", "normalized_summary.csv"))
stopifnot(identical(as.character(summary$component), original_summary$component),
          max(abs(as.matrix(summary[-1]) - as.matrix(original_summary[c("posterior_mean", "ci_lower", "ci_upper")]))) < 1e-12)
center <- data.frame(x = summary$posterior_mean[2] + 0.5 * summary$posterior_mean[3],
                     y = height * summary$posterior_mean[3])
kde <- MASS::kde2d(draws$x, draws$y, n = c(350, 350), lims = c(0, 1, 0, height))
density_at_draws <- function(x, y) {
  ix <- findInterval(x, kde$x, all.inside = TRUE)
  iy <- findInterval(y, kde$y, all.inside = TRUE)
  tx <- (x - kde$x[ix]) / (kde$x[ix + 1] - kde$x[ix])
  ty <- (y - kde$y[iy]) / (kde$y[iy + 1] - kde$y[iy])
  (1 - tx) * (1 - ty) * kde$z[cbind(ix, iy)] +
    tx * (1 - ty) * kde$z[cbind(ix + 1, iy)] +
    (1 - tx) * ty * kde$z[cbind(ix, iy + 1)] +
    tx * ty * kde$z[cbind(ix + 1, iy + 1)]
}
draws$posterior_density <- density_at_draws(draws$x, draws$y)
density_threshold <- unname(quantile(draws$posterior_density, 0.05))
draws$inside_approx_95_region <- draws$posterior_density >= density_threshold
density_grid <- expand.grid(x = kde$x, y = kde$y)
density_grid$density <- as.vector(kde$z)
inside_simplex <- density_grid$y >= 0 & density_grid$y <= height &
  density_grid$x >= density_grid$y / (2 * height) &
  density_grid$x <= 1 - density_grid$y / (2 * height)
density_grid$density[!inside_simplex] <- NA_real_ # Only the evaluation grid is masked.

curve_plot <- ggplot(curve, aes(x = latent_score)) +
  geom_ribbon(aes(ymin = fitness_lower, ymax = fitness_upper, fill = "95% GP CrI"),
              alpha = 0.2, colour = NA) +
  geom_line(aes(y = fitness_median, colour = "GP median"), linewidth = 0.65) +
  geom_segment(data = genotypes,
               aes(x = latent_lower, xend = latent_upper, y = fitness_median, yend = fitness_median),
               inherit.aes = FALSE, colour = "#D55E00", alpha = 0.22, linewidth = 0.25) +
  geom_segment(data = genotypes,
               aes(x = latent_median, xend = latent_median, y = fitness_lower, yend = fitness_upper),
               inherit.aes = FALSE, colour = "#D55E00", alpha = 0.3, linewidth = 0.25) +
  geom_point(data = genotypes, aes(x = latent_median, y = fitness_median, shape = "Genotype medians"),
             inherit.aes = FALSE, colour = "#D55E00", alpha = 0.85, size = 1.3) +
  geom_rug(data = tips, aes(x = latent_median), inherit.aes = FALSE, sides = "b",
           length = grid::unit(2.5, "pt"), linewidth = 0.25, alpha = 0.2, colour = "#333333") +
  scale_colour_manual(values = c("GP median" = "#0072B2"), name = NULL) +
  scale_fill_manual(values = c("95% GP CrI" = "#0072B2"), name = NULL) +
  scale_shape_manual(values = c("Genotype medians" = 16), name = NULL) +
  scale_x_continuous(limits = range(curve$latent_score), expand = expansion(mult = 0.015)) +
  scale_y_continuous(expand = expansion(mult = 0.035)) +
  guides(colour = guide_legend(order = 1), fill = guide_legend(order = 2),
         shape = guide_legend(order = 3)) +
  labs(x = "Latent additive score", y = "Fitness") +
  theme_classic(base_size = 8, base_family = "Helvetica") +
  theme(axis.text = element_text(size = 7, colour = "#333333"),
        axis.title = element_text(size = 8),
        axis.line = element_line(linewidth = 0.3, colour = "#444444"),
        axis.ticks = element_line(linewidth = 0.25, colour = "#444444"),
        legend.position = "bottom", legend.title = element_blank(),
        legend.text = element_text(size = 7), legend.spacing.x = grid::unit(2, "pt"),
        legend.key.width = grid::unit(11, "pt"), legend.key.height = grid::unit(10, "pt"),
        legend.box.spacing = grid::unit(4, "pt"),
        plot.margin = margin(7, 5, 5, 5, unit = "pt"))

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
  geom_contour_filled(data = density_grid, aes(x, y, z = density),
                      breaks = c(density_threshold, max(kde$z) + 1),
                      fill = "#94C3CF", alpha = 0.25, show.legend = FALSE, na.rm = TRUE) +
  geom_point(data = draws, aes(x, y), colour = "#185C7A", alpha = 0.22, size = 0.55) +
  geom_contour(data = density_grid, aes(x, y, z = density), breaks = density_threshold,
               colour = "#185C7A", linewidth = 0.5, na.rm = TRUE) +
  geom_path(data = triangle, aes(x, y), colour = "#555555", linewidth = 0.45) +
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

# Validate the actual layer coordinates, including both sets of full intervals.
curve_layers <- ggplot_build(curve_plot)$data
stopifnot(nrow(curve_layers[[1]]) == nrow(curve),
          max(abs(curve_layers[[1]]$ymin - curve$fitness_lower)) < 1e-12,
          max(abs(curve_layers[[1]]$ymax - curve$fitness_upper)) < 1e-12,
          max(abs(curve_layers[[2]]$y - curve$fitness_median)) < 1e-12,
          nrow(curve_layers[[3]]) == nrow(genotypes),
          max(abs(curve_layers[[3]]$x - genotypes$latent_lower)) < 1e-12,
          max(abs(curve_layers[[3]]$xend - genotypes$latent_upper)) < 1e-12,
          max(abs(curve_layers[[4]]$y - genotypes$fitness_lower)) < 1e-12,
          max(abs(curve_layers[[4]]$yend - genotypes$fitness_upper)) < 1e-12,
          nrow(curve_layers[[5]]) == nrow(genotypes),
          max(abs(curve_layers[[5]]$x - genotypes$latent_median)) < 1e-12,
          max(abs(curve_layers[[5]]$y - genotypes$fitness_median)) < 1e-12,
          nrow(curve_layers[[6]]) == nrow(tips))
ternary_layers <- ggplot_build(ternary_plot)$data
stopifnot(nrow(ternary_layers[[3]]) == nrow(draws),
          max(abs(ternary_layers[[3]]$x - draws$x)) < 1e-12,
          max(abs(ternary_layers[[3]]$y - draws$y)) < 1e-12,
          max(abs(ternary_layers[[6]]$x - center$x)) < 1e-12,
          max(abs(ternary_layers[[6]]$y - center$y)) < 1e-12)
combined <- (curve_plot + ternary_plot) + plot_layout(widths = c(1.1, 1)) +
  plot_annotation(tag_levels = "A") &
  theme(plot.tag = element_text(family = "Helvetica", size = 9, face = "bold"),
        plot.tag.position = c(0, 1), plot.tag.location = "plot")
width_mm <- 183
height_mm <- 90
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
stem <- file.path(out_dir, "gp_monotone_full_fitness_and_contributions")
grDevices::pdf(paste0(stem, ".pdf"), width = width_mm / 25.4, height = height_mm / 25.4,
               family = "Helvetica", useDingbats = FALSE)
print(combined)
# Measure resolved viewports rather than converting unresolved gtable null units.
grid::grid.force()
viewport_list <- grid::grid.ls(viewports = TRUE, grobs = FALSE, recursive = TRUE, print = FALSE)
panel_names <- unique(viewport_list$name[grepl("^panel(?:-[0-9]+)?\\.", viewport_list$name, perl = TRUE)])
if (length(panel_names) != 2) stop("Cannot identify the two rendered plot areas: ", paste(panel_names, collapse = ", "))
panels <- lapply(seq_along(panel_names), function(i) {
  grid::seekViewport(panel_names[i])
  lower <- grid::deviceLoc(grid::unit(0, "npc"), grid::unit(0, "npc"), valueOnly = TRUE)
  upper <- grid::deviceLoc(grid::unit(1, "npc"), grid::unit(1, "npc"), valueOnly = TRUE)
  grid::upViewport(0)
  tag_name <- unique(viewport_list$name[grepl(paste0("^tag-", i, "\\."), viewport_list$name)])
  stopifnot(length(tag_name) == 1)
  grid::seekViewport(tag_name)
  tag_anchor <- grid::deviceLoc(grid::unit(0, "npc"), grid::unit(1, "npc"), valueOnly = TRUE)
  grid::upViewport(0)
  list(id = LETTERS[i], bbox_pt = c(lower$x, lower$y, upper$x, upper$y) * 72,
       panel_label_anchor_pt = c(tag_anchor$x, tag_anchor$y) * 72)
})
stopifnot(abs(panels[[1]]$panel_label_anchor_pt[2] - panels[[2]]$panel_label_anchor_pt[2]) <= 1.5)
# Unlike the generic helper's unresolved gtable null units, these rectangles
# are measured on the rendered device. Enforce the 1.5 pt alignment gate here.
stopifnot(abs(panels[[1]]$bbox_pt[2] - panels[[2]]$bbox_pt[2]) <= 1.5,
          abs(panels[[1]]$bbox_pt[4] - panels[[2]]$bbox_pt[4]) <= 1.5,
          abs(diff(panels[[1]]$bbox_pt[c(2, 4)]) - diff(panels[[2]]$bbox_pt[c(2, 4)])) <= 1.5)
layout <- list(schema_version = 1L, backend = "r-patchwork-resolved-viewports",
               figure = list(width_pt = width_mm / 25.4 * 72, height_pt = height_mm / 25.4 * 72),
               panels = panels, row_groups = list(c("A", "B")),
               exemptions = list(list(panels = c("A", "B"), checks = "panel-width",
                 reason = "The Cartesian response panel is intentionally 1.1 times wider than the fixed-aspect equilateral ternary panel; both plot areas and panel labels align vertically.")))
jsonlite::write_json(layout, paste0(stem, ".panel-layout.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
grDevices::dev.off()
# Optional backend-neutral audit_panel_alignment.py report, in addition to the
# required in-process gate above. Set NATURE_PANEL_AUDIT to the auditor's path.
alignment_auditor <- Sys.getenv("NATURE_PANEL_AUDIT")
if (nzchar(alignment_auditor)) {
  stopifnot(file.exists(alignment_auditor))
  status <- system2(Sys.which("python3"), c(shQuote(alignment_auditor),
    shQuote(paste0(stem, ".panel-layout.json")), "--json-out",
    shQuote(paste0(stem, ".alignment-audit.json")), "--strict", "--require-panel-labels"))
  if (status != 0) stop("Backend-neutral panel alignment audit failed")
}
svglite::svglite(paste0(stem, ".svg"), width = width_mm / 25.4, height = height_mm / 25.4)
print(combined)
grDevices::dev.off()
ragg::agg_png(paste0(stem, ".png"), width = width_mm, height = height_mm, units = "mm", res = 300)
print(combined)
grDevices::dev.off()
ragg::agg_tiff(paste0(stem, ".tiff"), width = width_mm, height = height_mm, units = "mm", res = 600, compression = "lzw")
print(combined)
grDevices::dev.off()
write.csv(curve, paste0(stem, ".curve-source-data.csv"), row.names = FALSE)
write.csv(genotypes, paste0(stem, ".genotype-source-data.csv"), row.names = FALSE)
write.csv(draws, paste0(stem, ".contribution-source-data.csv"), row.names = FALSE)
write.csv(summary, paste0(stem, ".contribution-summary.csv"), row.names = FALSE)
metadata <- list(candidate = curve_metadata$candidate, backend = "R/ggplot2/patchwork",
                 width_mm = width_mm, height_mm = height_mm, n_panels = 2,
                 png_dpi = 300, tiff_dpi = 600, n_chains = curve_metadata$n_chains,
                 n_draws = nrow(draws), n_genotypes = nrow(genotypes), n_tips = nrow(tips),
                 n_excluded_draws = 0, n_excluded_genotypes = 0,
                 posterior_sha256 = contribution_metadata$posterior_sha256,
                 curve_estimator = "posterior median on fitness scale at fixed latent score",
                 genotype_estimator = "marginal posterior medians of paired latent-score and fitness draws",
                 intervals = "unchanged pointwise/marginal equal-tailed 95% credible intervals",
                 ternary_estimator = "posterior mean of shares normalized within each draw",
                 ternary_region = "approximate 95% KDE density region; 350 x 350 grid",
                 density_threshold = density_threshold,
                 empirical_region_coverage = mean(draws$inside_approx_95_region),
                 contribution_row_weighting = contribution_metadata$row_weighting,
                 n_contribution_rows = contribution_metadata$n_observations,
                 negative_roundoff_corrections = negative_roundoff_corrections,
                 branch_random_effects_included = FALSE,
                 posterior_summary = "median", contribution_statistics_preserved = TRUE, original_colours_preserved = TRUE,
                 title = FALSE, subtitle = FALSE, embedded_caption = FALSE,
                 figure_contract = list(question = "How does the fitted ST131 GP map relate latent score to fitness, and how is fitted log-fitness variation attributed?",
                   claim = "The full monotone GP permits a curved fitness response while assigning the largest mean relative unique contribution to latent interactions, with posterior uncertainty.",
                   panel_A_role = "fitted mapping and genotype uncertainty",
                   panel_B_role = "conditional component attribution and joint uncertainty"))
jsonlite::write_json(metadata, paste0(stem, ".metadata.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
message("Saved ", stem, ".pdf/.svg/.png/.tiff; posterior median curve and genotype coordinates; all original intervals retained")
