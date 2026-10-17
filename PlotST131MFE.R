# Plot background fitness (x) against MFE (y), one figure per mutation.
# Usage: Rscript PlotST131MFE.R ST131_MFE
# Optional NATURE_PANEL_AUDIT names the backend-neutral alignment audit script.

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})
for (package in c("jsonlite", "svglite", "ragg")) {
  if (!requireNamespace(package, quietly = TRUE)) stop(paste("Missing R package:", package))
}
args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 1) stop("Usage: Rscript PlotST131MFE.R [ST131_MFE_directory]")
input_dir <- if (length(args)) args[1] else "ST131_MFE"
out_dir <- file.path(input_dir, "figures", "by_mutation")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
summary_data <- read.csv(file.path(input_dir, "mutation_summaries.csv"), stringsAsFactors = FALSE)
pair_data <- read.csv(file.path(input_dir, "pair_summaries.csv"), stringsAsFactors = FALSE)
diagnostics <- read.csv(file.path(input_dir, "sampler_flags.csv"), stringsAsFactors = FALSE)
metadata <- jsonlite::fromJSON(file.path(input_dir, "analysis_metadata.json"))
features <- metadata$main_feature_names
families <- c("linear", "bspline_unconstrained", "bspline_monotone", "bspline_unimodal", "gp_monotone")
family_labels <- c(linear = "Linear", bspline_unconstrained = "Unconstrained spline",
                   bspline_monotone = "Monotone spline", bspline_unimodal = "Unimodal spline",
                   gp_monotone = "Monotone GP")
ordered_candidates <- unique(summary_data$candidate[order(match(summary_data$family, families),
                                                         match(summary_data$feature_set, c("reduced", "full")))])
if (!length(ordered_candidates)) stop("No MFE estimates found")
required_columns <- c("background_fitness_median", "estimated_mfe", "mfe_lower", "mfe_upper")
if (!all(required_columns %in% names(pair_data))) stop("Missing background fitness or paired MFE summaries")
if (any(!is.finite(as.matrix(pair_data[required_columns]))) ||
    any(as.matrix(pair_data[required_columns]) <= 0)) {
  stop("Background fitness and MFE summaries must be positive and finite")
}
if (any(pair_data$mfe_lower > pair_data$estimated_mfe | pair_data$estimated_mfe > pair_data$mfe_upper)) {
  stop("An MFE median lies outside its credible interval")
}
pair_data$background_status <- ifelse(tolower(as.character(pair_data$background_observed)) == "true",
                                     "Observed", "Unobserved")
theme_set(theme_classic(base_size = 7, base_family = "Helvetica") +
            theme(axis.line = element_line(linewidth = 0.3, colour = "#444444"),
                  axis.ticks = element_line(linewidth = 0.25, colour = "#444444"),
                  plot.title = element_text(size = 8, face = "bold"),
                  plot.subtitle = element_text(size = 6, colour = "#555555"),
                  axis.text = element_text(size = 6.5, colour = "#333333"),
                  axis.title = element_text(size = 7),
                  panel.grid.major.y = element_line(linewidth = 0.2, colour = "#E6E6E6"),
                  plot.margin = margin(3, 4, 3, 4, unit = "pt")))

background_limits <- function(values) {
  extent <- if (length(values)) range(values) else c(0.95, 1.05)
  padding <- max(diff(extent) * 0.06, 0.015)
  pmax(extent + c(-padding, padding), .Machine$double.eps)
}

make_panel <- function(candidate, mutation) {
  values <- summary_data[summary_data$candidate == candidate & summary_data$mutation == mutation, ]
  backgrounds <- pair_data[pair_data$candidate == candidate & pair_data$mutation == mutation, ]
  x_limits <- background_limits(backgrounds$background_fitness_median)
  log_extent <- log2(range(c(backgrounds$mfe_lower, backgrounds$mfe_upper, 1)))
  padding <- max(diff(log_extent) * 0.06, 0.04)
  y_limits <- 2 ^ (log_extent + c(-padding, padding))
  if (diff(log2(y_limits)) < 2) {
    y_breaks <- scales::breaks_extended(n = 4)(y_limits)
  } else {
    exponents <- seq(ceiling(log2(y_limits[1])), floor(log2(y_limits[2])))
    if (length(exponents) > 5) exponents <- unique(round(seq(min(exponents), max(exponents), length.out = 5)))
    y_breaks <- sort(unique(c(2 ^ exponents, 1)))
    # Keep the neutral tick without crowding it with a nearby power of two.
    y_breaks <- y_breaks[y_breaks == 1 | abs(log2(y_breaks)) >= diff(log2(y_limits)) / 6]
  }
  y_breaks <- y_breaks[y_breaks > 0 & y_breaks >= y_limits[1] & y_breaks <= y_limits[2]]
  x_breaks <- scales::breaks_extended(n = 4)(x_limits)
  x_breaks <- x_breaks[x_breaks >= x_limits[1] + diff(x_limits) * 0.05 &
                         x_breaks <= x_limits[2] - diff(x_limits) * 0.05]
  flags <- diagnostics[diagnostics$candidate == candidate, ]
  flag_text <- sprintf("%d chain%s; divergences %s; max site R-hat %s", flags$n_chains,
                       ifelse(flags$n_chains == 1, "", "s"),
                       ifelse(is.na(flags$divergences), "unavailable", flags$divergences),
                       ifelse(is.na(flags$max_site_effect_split_rhat), "unavailable",
                              sprintf("%.2f", flags$max_site_effect_split_rhat)))
  plot <- ggplot(backgrounds, aes(x = background_fitness_median, y = estimated_mfe)) +
    geom_hline(yintercept = 1, linetype = "dashed", linewidth = 0.35, colour = "#777777") +
    geom_vline(xintercept = if (x_limits[1] <= 1 && x_limits[2] >= 1) 1 else numeric(),
               linetype = "dashed", linewidth = 0.35, colour = "#777777") +
    geom_errorbar(aes(ymin = mfe_lower, ymax = mfe_upper), width = 0,
                  colour = "#185C7A", linewidth = 0.3, alpha = 0.28) +
    geom_point(aes(shape = background_status), colour = "#185C7A", size = 1.2, stroke = 0.35) +
    scale_shape_manual(values = c(Observed = 16, Unobserved = 1), guide = "none") +
    scale_x_continuous(limits = x_limits, breaks = x_breaks,
                       labels = scales::label_number(accuracy = 0.01)) +
    scale_y_continuous(trans = "log2", limits = y_limits, breaks = y_breaks,
                       labels = function(x) format(x, trim = TRUE, digits = 3,
                                                   scientific = diff(log2(y_limits)) > 12)) +
    labs(title = paste(family_labels[values$family[1]],
                       ifelse(values$feature_set[1] == "reduced", "| Reduced", "| Full")),
         subtitle = flag_text, x = "Background fitness", y = "MFE (log scale)")
  if (!nrow(backgrounds)) plot <- plot + annotate("text", x = mean(x_limits), y = 1,
                                                label = "No observed carriers", size = 2.5)
  built <- ggplot_build(plot)
  background_marks <- built$data[[4]]
  intervals <- built$data[[3]]
  if (nrow(background_marks) != nrow(backgrounds) ||
      any(!is.finite(background_marks$x) | !is.finite(background_marks$y)) ||
      any(is.na(background_marks$shape)) ||
      any(!is.finite(intervals$ymin) | !is.finite(intervals$ymax))) {
    stop(paste("Some genotype-background points or intervals cannot be drawn:", mutation, candidate))
  }
  if (!isTRUE(all.equal(background_marks$x, backgrounds$background_fitness_median)) ||
      !isTRUE(all.equal(background_marks$y, log2(backgrounds$estimated_mfe)))) {
    stop("The plotted axes do not match background fitness (x) and MFE (y)")
  }
  plot
}

# Measure actual drawn panel viewports rather than unresolved gtable null units.
require_patchwork_panel_alignment <- function(plot, stem, width_in, height_in) {
  probe <- tempfile(fileext = ".pdf")
  grDevices::pdf(probe, width = width_in, height = height_in, family = "Helvetica", useDingbats = FALSE)
  on.exit({grDevices::dev.off(); unlink(probe)}, add = TRUE)
  grob <- patchwork::patchworkGrob(plot)
  layout <- grob$layout[grepl("^panel-[0-9]+$", grob$layout$name), ]
  layout <- layout[order(layout$t, layout$l), ]
  if (nrow(layout) < 2) return(invisible(NULL))
  grid::grid.newpage()
  grid::grid.draw(grob)
  grid::grid.force()
  viewports <- grid::grid.ls(viewports = TRUE, grobs = FALSE, print = FALSE)$name
  panels <- lapply(seq_len(nrow(layout)), function(index) {
    row <- layout[index, ]
    matching <- grep(paste0("^", row$name, "\\."), viewports, value = TRUE)
    if (length(matching) != 1) stop(paste("Cannot measure panel viewport", row$name))
    grid::seekViewport(matching)
    bottom <- grid::deviceLoc(grid::unit(0, "npc"), grid::unit(0, "npc"), valueOnly = TRUE)
    top <- grid::deviceLoc(grid::unit(1, "npc"), grid::unit(1, "npc"), valueOnly = TRUE)
    grid::upViewport(0)
    list(id = letters[index], bbox_pt = 72 * c(bottom$x, bottom$y, top$x, top$y),
         grid_id = "mfe-grid", row_start = row$t - 1L, row_stop = row$b,
         col_start = row$l - 1L, col_stop = row$r)
  })
  manifest_path <- paste0(stem, ".alignment-layout.json")
  jsonlite::write_json(list(schema_version = 1L, backend = "r-patchwork",
                           figure = list(width_pt = width_in * 72, height_pt = height_in * 72),
                           panels = panels, exemptions = list()), manifest_path,
                      auto_unbox = TRUE, pretty = TRUE, digits = NA)
  audit_script <- Sys.getenv("NATURE_PANEL_AUDIT", "")
  if (nzchar(audit_script)) {
    result <- system2(Sys.which("python3"), c(shQuote(audit_script), shQuote(manifest_path),
                                            "--json-out", shQuote(paste0(stem, ".alignment.json")), "--strict"))
    if (result != 0) stop("Panel alignment failed; fix layout before exporting")
  }
}

save_figure <- function(plot, name, width_mm, height_mm, multipanel = FALSE) {
  stem <- file.path(out_dir, name)
  width_in <- width_mm / 25.4
  height_in <- height_mm / 25.4
  if (multipanel) require_patchwork_panel_alignment(plot, stem, width_in, height_in)
  # Native pdf preserves editable Helvetica text without the unavailable
  # XQuartz libraries required by cairo_pdf on this machine.
  grDevices::pdf(paste0(stem, ".pdf"), width = width_in, height = height_in,
                 family = "Helvetica", useDingbats = FALSE)
  print(plot)
  grDevices::dev.off()
  svglite::svglite(paste0(stem, ".svg"), width = width_in, height = height_in)
  print(plot)
  grDevices::dev.off()
  ragg::agg_png(paste0(stem, ".png"), width = width_mm, height = height_mm, units = "mm", res = 300)
  print(plot)
  grDevices::dev.off()
  message("Saved ", stem, ".pdf/.svg/.png")
}

caption <- paste0("Each point is one carrier genotype: x = posterior median background fitness; y = median paired fitness ratio.\n",
                  "Vertical bars: 95% credible intervals for MFE; filled/open points: observed/unobserved background.\n",
                  "Dashed lines: fitness or MFE = 1. Axis ranges vary by model to show all points and full intervals; y is log-scaled.")
mutation_figures <- list()
figure_records <- list()
panel_records <- list()
single_candidate <- length(ordered_candidates) == 1L
figure_width_mm <- if (single_candidate) 183 else 210
figure_height_mm <- if (single_candidate) 160 else 285
for (index in seq_along(features)) {
  mutation <- features[index]
  message(sprintf("[%d/%d] Plotting mutation: %s", index, length(features), mutation))
  backgrounds <- pair_data[pair_data$mutation == mutation, ]
  panels <- lapply(ordered_candidates, make_panel, mutation = mutation)
  values <- summary_data[summary_data$mutation == mutation, ]
  figure <- wrap_plots(panels, ncol = if (single_candidate) 1 else 2) +
    plot_annotation(title = paste0("ST131 | ", sub("_AMR$", "", mutation)),
                    subtitle = sprintf("%d sampled carriers | %d distinct genetic backgrounds | %d candidate model%s",
                                       values$n_carriers[1], values$n_backgrounds[1], length(panels),
                                       if (single_candidate) "" else "s"),
                    caption = caption,
                    theme = theme(plot.title = element_text(size = 11, face = "bold"),
                                  plot.subtitle = element_text(size = 8),
                                  plot.caption = element_text(size = 6.5, hjust = 0),
                                  plot.margin = margin(6, 6, 6, 6, unit = "pt")))
  name <- paste0("mfe_background_", mutation)
  save_figure(figure, name, width_mm = figure_width_mm, height_mm = figure_height_mm, multipanel = TRUE)
  mutation_figures[[mutation]] <- figure
  figure_records[[index]] <- data.frame(mutation = mutation, pdf = paste0(name, ".pdf"),
                                         svg = paste0(name, ".svg"), png = paste0(name, ".png"),
                                         n_carriers = values$n_carriers[1],
                                         n_backgrounds = values$n_backgrounds[1])
  panel_records[[index]] <- do.call(rbind, lapply(ordered_candidates, function(candidate) {
    rows <- backgrounds[backgrounds$candidate == candidate, ]
    x_limits <- background_limits(rows$background_fitness_median)
    data.frame(mutation = mutation, candidate = candidate, n_points = nrow(rows),
               n_unobserved_backgrounds = sum(rows$background_status == "Unobserved"),
               x_statistic = "posterior median background fitness",
               y_statistic = "posterior median paired MFE",
               uncertainty = "full equal-tailed 95% MFE credible interval",
               x_min = x_limits[1], x_max = x_limits[2],
               all_points_and_intervals_drawn = TRUE)
  }))
}
grDevices::pdf(file.path(out_dir, "all_mutations.pdf"), width = figure_width_mm / 25.4, height = figure_height_mm / 25.4,
               family = "Helvetica", useDingbats = FALSE)
for (figure in mutation_figures) print(figure)
grDevices::dev.off()
write.csv(do.call(rbind, figure_records), file.path(out_dir, "figure_index.csv"), row.names = FALSE)
write.csv(do.call(rbind, panel_records), file.path(out_dir, "panel_audit.csv"), row.names = FALSE)
message("Saved combined PDF: ", file.path(out_dir, "all_mutations.pdf"))
writeLines(c(sprintf("Figure question: How does each mutation's MFE vary with background fitness under the %d supplied candidate model(s)?", length(ordered_candidates)),
             "Archetype: scatter-plot grid; each figure contains one mutation and one panel per candidate model.",
             "X: posterior median predicted fitness of the genotype with the selected feature set to zero.",
             "Y: posterior median of draw-paired mutant/background fitness ratios; logarithmic axis.",
             "Uncertainty: complete equal-tailed 95% credible intervals for MFE in every panel.",
             "The x coordinate is a posterior point estimate; background-fitness uncertainty is not displayed.",
             "One point per distinct carrier genotype; no jitter, smoothing, or connections between genotypes.",
             "Filled points: counterfactual background observed among sampled tips; open points: unobserved background.",
             "Axis ranges vary by model to retain every point and interval in full; compare numeric axis labels between panels.",
             "No model averaging, convergence filtering, clipping, or posterior draw subsampling is applied.",
             "Source data: pair_summaries.csv, mutation_summaries.csv, sampler_flags.csv, analysis_metadata.json.",
             "All genotype/background points and credible intervals are validated after ggplot's axis transformations.",
             "Conditional fitted effects are not experimentally measured causal effects; check the displayed sampler diagnostics."),
           file.path(out_dir, "figure_notes.txt"))
