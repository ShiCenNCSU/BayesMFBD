#!/usr/bin/env Rscript
# Plot all full-GP mutation/background MFE estimates, one panel per mutation.
# Run: Rscript PlotST131GPAllMFE.R [ST131_MFE_directory] [output_directory]
# x: posterior mean background fitness; y: posterior mean paired fitness ratio.
# All genotype-specific estimates and full 95% credible intervals are retained.
# Publication artwork has no global heading or embedded caption; the figure
# legend and carrier/background count table are exported separately.
# Optional NATURE_PANEL_AUDIT runs the backend-neutral measured-layout audit.

suppressPackageStartupMessages(library(ggplot2))
for (package in c("jsonlite", "svglite", "ragg", "scales")) {
  if (!requireNamespace(package, quietly = TRUE)) stop("Required R package unavailable: ", package)
}
args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 2) stop("Usage: Rscript PlotST131GPAllMFE.R [input_directory] [output_directory]")
input_dir <- if (length(args)) args[1] else "ST131_MFE"
out_dir <- if (length(args) > 1) args[2] else file.path(input_dir, "figures", "gp_full_all_mutations")
candidate <- "gp_monotone_full"
# The dedicated GP mean summaries preserve the existing candidate-comparison
# tables while making the plotted posterior estimator explicit.
summary_dir <- file.path(input_dir, "gp_full_mean")
if (!file.exists(file.path(summary_dir, "pair_summaries.csv"))) summary_dir <- input_dir
all_pairs <- read.csv(file.path(summary_dir, "pair_summaries.csv"), stringsAsFactors = FALSE,
                      colClasses = c(mutant_genotype = "character", background_genotype = "character"))
metadata <- jsonlite::fromJSON(file.path(summary_dir, "analysis_metadata.json"))
flags <- read.csv(file.path(summary_dir, "sampler_flags.csv"), stringsAsFactors = FALSE)
pairs <- all_pairs[all_pairs$candidate == candidate, , drop = FALSE]
if (!nrow(pairs)) stop("No gp_monotone_full MFE estimates found")
if (any(c(pairs$family != "gp_monotone", pairs$feature_set != "full"))) stop("Candidate is not the full monotone GP")
features <- metadata$main_feature_names
if (!setequal(unique(pairs$mutation), features)) stop("The GP fit does not contain estimates for every selected AMR feature")
required <- c("background_fitness_mean", "posterior_mean", "mfe_lower", "mfe_upper")
if (!all(required %in% names(pairs)) || !all(pairs$point_estimator == "posterior_mean")) {
  stop("Recompute GP mean summaries: python ST131MFE.py --candidate gp_monotone_full --point-estimator mean --output-dir ST131_MFE/gp_full_mean")
}
values <- as.matrix(pairs[required])
if (any(!is.finite(values)) || any(values <= 0)) stop("Fitness and MFE values must be finite and positive")
if (any(pairs$mfe_lower > pairs$mfe_upper)) stop("An MFE credible interval has reversed bounds")
if (max(abs(pairs$estimated_mfe - pairs$posterior_mean)) > 1e-12) stop("The primary MFE estimate is not the posterior mean")
if (anyDuplicated(pairs[c("mutation", "mutant_genotype", "background_genotype")])) {
  stop("Duplicate mutation/background estimates found")
}
observed <- tolower(as.character(pairs$background_observed))
if (!all(observed %in% c("true", "false"))) stop("Invalid background-observation flags")
pairs$background_status <- factor(ifelse(observed == "true", "Observed", "Unobserved"),
                                  levels = c("Observed", "Unobserved"))
pairs$mutation <- factor(pairs$mutation, levels = features)
fit <- flags[flags$candidate == candidate, , drop = FALSE]
if (nrow(fit) != 1) stop("Expected one diagnostic row for gp_monotone_full")
mutation_summaries <- read.csv(file.path(summary_dir, "mutation_summaries.csv"), stringsAsFactors = FALSE)
counts <- mutation_summaries[mutation_summaries$candidate == candidate, , drop = FALSE]
if (nrow(counts) != length(features) || anyDuplicated(counts$mutation) || !setequal(counts$mutation, features)) {
  stop("Expected one carrier-count summary per fitted feature")
}
counts <- counts[match(features, counts$mutation), , drop = FALSE]
if (!all(counts$n_backgrounds == as.integer(table(pairs$mutation)))) stop("Summary background counts differ from plotted pairs")

# Restore the original colour assigned to each mutation/AMR feature.
# Filled/open symbols identify observed/counterfactual background genotypes.
palette <- c(gyrAS83L_AMR = "#0072B2", parCE84V_AMR = "#D55E00",
             gyrAD87NparCS80I_AMR = "#009E73", parEI529L_AMR = "#CC79A7",
             blaCTXM27_AMR = "#E69F00", tet_AMR = "#56B4E9",
             mdtM_AMR = "#333333", glpTE448K_AMR = "#5E3C99",
             blaCTXM15_AMR = "#8C6D31", aac3_AMR = "#A63658",
             aac6_AMR = "#536B85", sul_AMR = "#287D7D",
             aac6IbcrblaOXAOH_AMR = "#827D42")
if (!all(features %in% names(palette))) stop("Missing original colour for a fitted feature")
palette <- palette[features]
feature_labels <- setNames(sub("_AMR$", "", features), features)
panel_labels <- setNames(sprintf("%s  %s", letters[seq_along(features)], unname(feature_labels)), features)
x_extent <- range(pairs$background_fitness_mean)
x_padding <- max(diff(x_extent) * 0.05, 0.015)
x_limits <- x_extent + c(-x_padding, x_padding)
log_extent <- log2(range(c(pairs$mfe_lower, pairs$mfe_upper, pairs$posterior_mean, 1)))
log_padding <- max(diff(log_extent) * 0.06, 0.035)
y_limits <- 2 ^ (log_extent + c(-log_padding, log_padding))
y_breaks <- scales::breaks_extended(n = 7)(y_limits)
y_breaks <- sort(unique(c(y_breaks, 1)))
y_breaks <- y_breaks[y_breaks > y_limits[1] & y_breaks < y_limits[2]]

theme_set(theme_classic(base_size = 8, base_family = "Helvetica") +
            theme(axis.line = element_line(linewidth = 0.3, colour = "#444444"),
                  axis.ticks = element_line(linewidth = 0.25, colour = "#444444"),
                  axis.text = element_text(size = 6.5, colour = "#333333"),
                  axis.title = element_text(size = 8),
                  panel.grid.major.y = element_line(linewidth = 0.2, colour = "#ECECEC"),
                  plot.title = element_blank(),
                  plot.subtitle = element_blank(),
                  legend.title = element_text(size = 7, face = "bold"),
                  legend.text = element_text(size = 7),
                  legend.key.height = grid::unit(11, "pt"),
                  legend.key.width = grid::unit(12, "pt"),
                  legend.spacing.y = grid::unit(4, "pt"),
                  legend.box.spacing = grid::unit(7, "pt"),
                  plot.caption = element_blank(),
                  strip.background = element_blank(),
                  strip.text = element_text(size = 8, face = "bold", hjust = 0, margin = margin(3, 0, 4, 0)),
                  panel.spacing = grid::unit(8, "pt"),
                  legend.position = "bottom",
                  plot.margin = margin(6, 6, 6, 6, unit = "pt")))
plot <- ggplot(pairs, aes(x = background_fitness_mean, y = posterior_mean, colour = mutation)) +
  geom_hline(yintercept = 1, colour = "#777777", linewidth = 0.35, linetype = "dashed") +
  geom_errorbar(aes(ymin = mfe_lower, ymax = mfe_upper), width = 0, alpha = 0.35,
                linewidth = 0.25, show.legend = FALSE) +
  geom_point(aes(shape = background_status), size = 1.1, stroke = 0.35, alpha = 0.95) +
  scale_colour_manual(values = palette, breaks = features, labels = feature_labels,
                       name = "Mutation / AMR feature", drop = FALSE) +
  scale_shape_manual(values = c(Observed = 16, Unobserved = 1), name = "Background genotype", drop = FALSE) +
  scale_x_continuous(limits = x_limits, breaks = scales::breaks_extended(n = 5),
                      labels = scales::label_number(accuracy = 0.01), expand = expansion(mult = 0)) +
  scale_y_continuous(trans = "log2", limits = y_limits, breaks = y_breaks,
                      labels = scales::label_number(accuracy = 0.01), expand = expansion(mult = 0)) +
  guides(colour = "none", shape = guide_legend(nrow = 1, override.aes = list(colour = "#444444", alpha = 1, size = 2))) +
  facet_wrap(vars(mutation), ncol = 3, scales = "fixed", labeller = as_labeller(panel_labels)) +
  labs(title = NULL, subtitle = NULL, caption = NULL,
       x = "Background fitness", y = "MFE (log scale)")

# Verify every point and interval is preserved in the intended mutation panel.
built <- ggplot_build(plot)
marks <- built$data[[3]]
intervals <- built$data[[2]]
stopifnot(nrow(marks) == nrow(pairs), nrow(intervals) == nrow(pairs),
          all(is.finite(marks$x)), all(is.finite(marks$y)), all(!is.na(marks$colour)),
          all(is.finite(intervals$ymin)), all(is.finite(intervals$ymax)),
          max(abs(marks$x - pairs$background_fitness_mean)) < 1e-12,
          max(abs(marks$y - log2(pairs$posterior_mean))) < 1e-12,
          max(abs(intervals$ymin - log2(pairs$mfe_lower))) < 1e-12,
          max(abs(intervals$ymax - log2(pairs$mfe_upper))) < 1e-12)
facet_layout <- built$layout$layout
stopifnot(nrow(facet_layout) == length(features))
for (i in seq_len(nrow(facet_layout))) {
  feature <- as.character(facet_layout$mutation[i])
  panel <- as.integer(facet_layout$PANEL[i])
  expected <- pairs[pairs$mutation == feature, , drop = FALSE]
  actual <- marks[as.integer(marks$PANEL) == panel, , drop = FALSE]
  actual_intervals <- intervals[as.integer(intervals$PANEL) == panel, , drop = FALSE]
  stopifnot(nrow(actual) == nrow(expected), nrow(actual_intervals) == nrow(expected),
            max(abs(actual$x - expected$background_fitness_mean)) < 1e-12,
            max(abs(actual$y - log2(expected$posterior_mean))) < 1e-12)
}

# Measure actual drawn facet viewports at the final export dimensions. Null
# gtable units cannot establish plot widths, so use deviceLoc after drawing.
require_facet_panel_alignment <- function(plot, facet_layout, stem, width_in, height_in) {
  probe <- tempfile(fileext = ".pdf")
  grDevices::pdf(probe, width = width_in, height = height_in, family = "Helvetica", useDingbats = FALSE)
  on.exit({grDevices::dev.off(); unlink(probe)}, add = TRUE)
  grob <- ggplotGrob(plot)
  grid::grid.newpage()
  grid::grid.draw(grob)
  grid::grid.force()
  viewports <- grid::grid.ls(viewports = TRUE, grobs = FALSE, print = FALSE)$name
  panels <- lapply(seq_len(nrow(facet_layout)), function(index) {
    cell <- facet_layout[index, ]
    name <- sprintf("panel-%d-%d", cell$COL, cell$ROW)
    matching <- grep(paste0("^", name, "\\."), viewports, value = TRUE)
    if (length(matching) != 1) stop("Cannot measure facet viewport: ", name)
    grid::seekViewport(matching)
    bottom <- grid::deviceLoc(grid::unit(0, "npc"), grid::unit(0, "npc"), valueOnly = TRUE)
    top <- grid::deviceLoc(grid::unit(1, "npc"), grid::unit(1, "npc"), valueOnly = TRUE)
    grid::upViewport(0)
    list(id = letters[index], bbox_pt = 72 * c(bottom$x, bottom$y, top$x, top$y),
         grid_id = "gp-mutation-facets", row_start = cell$ROW - 1L, row_stop = cell$ROW,
         col_start = cell$COL - 1L, col_stop = cell$COL)
  })
  boxes <- do.call(rbind, lapply(panels, function(panel) panel$bbox_pt))
  tolerance <- 1.5
  if (diff(range(boxes[, 3] - boxes[, 1])) > tolerance ||
      diff(range(boxes[, 4] - boxes[, 2])) > tolerance) stop("Facet widths or heights differ by more than 1.5 pt")
  for (row in unique(facet_layout$ROW)) {
    ids <- which(facet_layout$ROW == row)
    if (diff(range(boxes[ids, 2])) > tolerance || diff(range(boxes[ids, 4])) > tolerance) stop("Facet row edges do not align")
  }
  for (column in unique(facet_layout$COL)) {
    ids <- which(facet_layout$COL == column)
    if (diff(range(boxes[ids, 1])) > tolerance || diff(range(boxes[ids, 3])) > tolerance) stop("Facet column edges do not align")
  }
  manifest_path <- paste0(stem, ".alignment-layout.json")
  jsonlite::write_json(list(schema_version = 1L, backend = "r-ggplot2-facets",
                           figure = list(width_pt = width_in * 72, height_pt = height_in * 72),
                           panels = panels, exemptions = list()), manifest_path,
                      auto_unbox = TRUE, pretty = TRUE, digits = NA)
  audit_script <- Sys.getenv("NATURE_PANEL_AUDIT", "")
  if (nzchar(audit_script)) {
    status <- system2(Sys.which("python3"), c(shQuote(audit_script), shQuote(manifest_path),
                                             "--json-out", shQuote(paste0(stem, ".alignment.json")), "--strict"))
    if (status != 0) stop("Facet alignment failed; fix layout before export")
  } else {
    jsonlite::write_json(list(verdict = "PASS", backend = "native_R_measured_facet_edges",
                             n_panels = nrow(facet_layout), tolerance_pt = tolerance),
                        paste0(stem, ".alignment.json"), auto_unbox = TRUE, pretty = TRUE)
  }
}
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
stem <- file.path(out_dir, "mfe_gp_monotone_full_mutation_panels")
width_mm <- 183
height_mm <- 230
require_facet_panel_alignment(plot, facet_layout, stem, width_mm / 25.4, height_mm / 25.4)
# Native PDF retains editable Helvetica text without the local XQuartz/Cairo dependency.
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
ragg::agg_tiff(paste0(stem, ".tiff"), width = width_mm, height = height_mm, units = "mm", res = 600,
               compression = "lzw")
print(plot)
grDevices::dev.off()
write.csv(pairs, file.path(out_dir, "source_data.csv"), row.names = FALSE)
audit <- data.frame(panel = letters[seq_along(features)], mutation = features,
                    row = facet_layout$ROW, column = facet_layout$COL,
                    plotted_pairs = as.integer(table(pairs$mutation)),
                    all_points_preserved = TRUE, all_intervals_preserved = TRUE,
                    colour = unname(palette), label = unname(feature_labels))
write.csv(audit, file.path(out_dir, "mutation_audit.csv"), row.names = FALSE)
panel_counts <- data.frame(panel = audit$panel, mutation = features,
                           sampled_carriers = counts$n_carriers,
                           distinct_backgrounds = counts$n_backgrounds,
                           plotted_pairs = audit$plotted_pairs)
write.csv(panel_counts, file.path(out_dir, "panel_counts.csv"), row.names = FALSE)
legend_text <- c("# Mutational fitness effects across genetic backgrounds under the full monotone GP",
                "",
                sprintf("Panels a–%s show the %d encoded ST131 mutation/AMR features. Each point represents a distinct sampled carrier genotype with its selected feature removed to define the background; pairwise interactions are recomputed. The x coordinate is posterior mean background fitness, and the y coordinate is the posterior mean draw-paired mutant/background fitness ratio (MFE). The MFE mean averages ratios within posterior draws, rather than taking a ratio of separate mean fitness predictions. Vertical bars show full equal-tailed 95%% credible intervals for MFE from %s retained draws across %d chain(s). All panels share identical axis ranges; the y-axis is logarithmic.",
                        tail(audit$panel, 1), length(features), format(fit$n_chains * fit$draws_per_chain, big.mark = ","), fit$n_chains),
                "",
                "Filled points denote backgrounds observed among sampled tips; open points denote unobserved counterfactual backgrounds. The dashed line marks MFE = 1. Points are unweighted, with no jitter, smoothing, or connecting lines. Uncertainty in the x coordinate is not displayed. Some labels represent combined encoded indicators, for which the MFE applies to the complete feature.",
                "",
                sprintf("The analysis includes %d sampled tree tips and %d distinct mutation/background pairs. Sampled carriers and distinct plotted backgrounds are different counts. The table below defines both; reconstructed ancestral nodes are excluded from these tip-based counts.", metadata$n_sampled_tips, nrow(pairs)),
                "",
                "| Panel | Mutation/AMR feature | Sampled carriers | Distinct backgrounds |",
                "|---|---|---:|---:|")
legend_text <- c(legend_text, sprintf("| %s | %s | %d | %d |", panel_counts$panel,
                                     unname(feature_labels), panel_counts$sampled_carriers,
                                     panel_counts$distinct_backgrounds))
if (fit$n_chains == 1) legend_text <- c(legend_text, "", "These are preliminary conditional posterior estimates from one chain; publication-quality artwork does not establish posterior convergence.")
writeLines(legend_text, paste0(stem, ".legend.md"), useBytes = TRUE)
posterior_record <- metadata$candidates[metadata$candidates$candidate == candidate, , drop = FALSE]
report <- list(candidate = candidate, n_features = length(features), n_input_pairs = nrow(pairs),
               n_panels = length(features), n_columns = 3, n_rows = ceiling(length(features) / 3),
               layout = "one_panel_per_mutation", shared_x_limits = x_limits, shared_y_limits = y_limits,
               n_plotted_pairs = nrow(marks), n_plotted_intervals = nrow(intervals), n_excluded_pairs = 0,
               x = "posterior_mean_background_fitness", y = "posterior_mean_draw_paired_fitness_ratio",
               summary_directory = summary_dir,
               y_scale = "log2", interval = "full_equal_tailed_95_percent_credible_interval_on_MFE",
               x_uncertainty_shown = FALSE, point_weighting = "one_point_per_distinct_carrier_genotype_and_feature",
               genotype_filter = "sampled_tips_only; as in ST131MFE.py", line_connections = FALSE, smoothing = FALSE,
               n_chains = fit$n_chains, draws_per_chain = fit$draws_per_chain, divergences = fit$divergences,
               posterior_sha256 = posterior_record$sha256, features = features,
               width_mm = width_mm, height_mm = height_mm, png_dpi = 300, tiff_dpi = 600,
               figure_title = FALSE, figure_subtitle = FALSE, embedded_caption = FALSE,
               publication_style = "original_mutation_colours; unshaded_mutation_labels; external_legend_and_counts",
               backend = "R/ggplot2", R_version = R.version.string,
               figure_contract = list(question = "How do mutation MFE estimates differ across observed genetic backgrounds in the full GP model?",
                                      claim = "Mutation panels display genotype-specific effects and posterior uncertainty on shared axes",
                                      archetype = "quantitative grid; one panel per mutation",
                                      alignment = "measured_facet_viewports; strict_1.5_pt_gate"))
jsonlite::write_json(report, paste0(stem, ".metadata.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
writeLines(c("All 644 genotype-specific pairs for the 13 full-GP AMR features are retained; check metadata for counts on new data.",
             "Each feature has a separately labelled panel with identical x and y ranges; labels contain only panel letters and feature names.",
             "The original mutation colour palette is retained; filled/open symbols identify observed/unobserved backgrounds.",
             "No global title, subtitle, or caption is embedded in the artwork; a standalone legend and panel-count table are exported.",
             "Full 95% MFE intervals are shown without clipping. Panel areas are measured and aligned before export.",
             "The y axis is logarithmic. No points or intervals are clipped, jittered, smoothed, connected, or carrier-count weighted.",
             "Background fitness and MFE use posterior mean point estimates; x-coordinate uncertainty is not shown.",
             "Some feature names encode combined indicators; interpret estimates for the complete encoded feature.",
             "One-chain conditional posterior results remain preliminary."), file.path(out_dir, "figure_notes.txt"))
message("Saved ", stem, ".pdf/.svg/.png/.tiff and figure legend; all ", nrow(pairs), " pairs across ", length(features), " mutation panels shown")
