#!/usr/bin/env Rscript
# R-only plotting of interaction-coefficient ablations, using every supplied row.
# Run: Rscript PlotST131GPInteractionMFE.R SUMMARY_DIRECTORY [OUTPUT_DIRECTORY]
suppressPackageStartupMessages(library(ggplot2))
for (package in c("jsonlite", "svglite", "ragg", "scales")) {
  if (!requireNamespace(package, quietly = TRUE)) stop("Missing R package: ", package)
}
args <- commandArgs(trailingOnly = TRUE)
if (!length(args) %in% c(1, 2)) stop("Supply interaction summary directory and optional output directory")
input_dir <- args[1]
out_dir <- if (length(args) == 2) args[2] else file.path(input_dir, "figures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
metadata <- jsonlite::fromJSON(file.path(input_dir, "analysis_metadata.json"))
pairs <- read.csv(file.path(input_dir, "pair_summaries.csv"), colClasses = c(carrier_genotype = "character"))
summary <- read.csv(file.path(input_dir, "interaction_summaries.csv"))
features <- metadata$interaction_names
if (metadata$model != "gp_monotone" || metadata$feature_set != "full") stop("Expected full monotone GP")
if (nrow(summary) != metadata$n_interactions || anyDuplicated(summary$interaction) ||
    !identical(as.character(summary$interaction), features)) stop("Interaction summary order/count mismatch")
if (nrow(pairs) != metadata$n_interaction_background_pairs ||
    anyDuplicated(pairs[c("interaction", "carrier_genotype")])) stop("Background count/uniqueness mismatch")
if (any(pairs$posterior_draw_count != metadata$n_total_draws) ||
    any(summary$posterior_draw_count != metadata$n_total_draws)) stop("Posterior draws were omitted")
if (any(summary$status != "estimated")) stop("This figure requires sampled co-carriers for every fitted interaction")
required <- c("background_fitness_mean", "posterior_mean", "mfe_lower", "mfe_upper")
if (any(!is.finite(as.matrix(pairs[required]))) || any(as.matrix(pairs[required]) <= 0) ||
    any(!is.finite(as.matrix(summary[c("posterior_mean", "mfe_lower", "mfe_upper")])) ) ||
    any(as.matrix(summary[c("posterior_mean", "mfe_lower", "mfe_upper")]) <= 0)) {
  stop("Logarithmic axes require finite positive means and credible bounds")
}
stopifnot(all(pairs$mfe_lower <= pairs$mfe_upper), all(summary$mfe_lower <= summary$mfe_upper),
          setequal(unique(pairs$interaction), features),
          all(summary$n_backgrounds == as.integer(table(factor(pairs$interaction, levels = features)))))
palette <- c(gyrAS83L_AMR = "#0072B2", parCE84V_AMR = "#D55E00",
             gyrAD87NparCS80I_AMR = "#009E73", parEI529L_AMR = "#CC79A7",
             blaCTXM27_AMR = "#E69F00", tet_AMR = "#56B4E9",
             mdtM_AMR = "#333333", glpTE448K_AMR = "#5E3C99",
             blaCTXM15_AMR = "#8C6D31", aac3_AMR = "#A63658", aac6_AMR = "#536B85",
             sul_AMR = "#287D7D", aac6IbcrblaOXAOH_AMR = "#827D42")
if (!all(pairs$mutation_1 %in% names(palette))) stop("Missing mutation colour")
display <- function(x) sub("_AMR$", "", x)
panel_labels <- setNames(sprintf("%02d  %s \u00d7\n%s", seq_along(features),
                                 display(summary$mutation_1), display(summary$mutation_2)), features)
theme_set(theme_classic(base_size = 8, base_family = "Helvetica") +
            theme(axis.line = element_line(linewidth = .3, colour = "#444444"),
                  axis.ticks = element_line(linewidth = .25, colour = "#444444"),
                  axis.text = element_text(size = 6.5, colour = "#333333"),
                  axis.title = element_text(size = 8),
                  panel.grid.major.y = element_line(linewidth = .2, colour = "#ECECEC"),
                  strip.background = element_blank(),
                  strip.text = element_text(size = 7, face = "bold", hjust = 0, lineheight = .95,
                                            margin = margin(3, 0, 4, 0)),
                  panel.spacing = grid::unit(9, "pt"), legend.position = "none",
                  plot.title = element_blank(), plot.subtitle = element_blank(), plot.caption = element_blank(),
                  plot.margin = margin(6, 6, 6, 6, "pt")))

# Adapted from the existing ST131 MFE plotter: measure drawn facet viewports,
# then apply the same 1.5-point alignment contract to the final rectangles.
require_facet_panel_alignment <- function(plot, facet_layout, stem, width_mm, height_mm) {
  probe <- tempfile(fileext = ".pdf")
  grDevices::pdf(probe, width = width_mm / 25.4, height = height_mm / 25.4,
                 family = "Helvetica", useDingbats = FALSE)
  on.exit({grDevices::dev.off(); unlink(probe)}, add = TRUE)
  grid::grid.newpage()
  grid::grid.draw(ggplotGrob(plot))
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
    list(id = paste0("p", index), bbox_pt = 72 * c(bottom$x, bottom$y, top$x, top$y),
         grid_id = "interaction-mfe-facets", row_start = cell$ROW - 1L, row_stop = cell$ROW,
         col_start = cell$COL - 1L, col_stop = cell$COL)
  })
  boxes <- do.call(rbind, lapply(panels, function(panel) panel$bbox_pt))
  if (diff(range(boxes[, 3] - boxes[, 1])) > 1.5 || diff(range(boxes[, 4] - boxes[, 2])) > 1.5) {
    stop("Facet dimensions differ by more than 1.5 pt")
  }
  for (row in unique(facet_layout$ROW)) {
    ids <- which(facet_layout$ROW == row)
    if (diff(range(boxes[ids, 2])) > 1.5 || diff(range(boxes[ids, 4])) > 1.5) stop("Facet row misalignment")
  }
  for (column in unique(facet_layout$COL)) {
    ids <- which(facet_layout$COL == column)
    if (diff(range(boxes[ids, 1])) > 1.5 || diff(range(boxes[ids, 3])) > 1.5) stop("Facet column misalignment")
  }
  manifest <- paste0(stem, ".alignment-layout.json")
  jsonlite::write_json(list(schema_version = 1L, backend = "r-ggplot2-facets",
                           figure = list(width_pt = width_mm / 25.4 * 72, height_pt = height_mm / 25.4 * 72),
                           panels = panels, exemptions = list()), manifest, auto_unbox = TRUE, pretty = TRUE, digits = NA)
  auditor <- Sys.getenv("NATURE_PANEL_AUDIT", "")
  if (nzchar(auditor)) {
    status <- system2(Sys.which("python3"), c(shQuote(auditor), shQuote(manifest), "--json-out",
                                              shQuote(paste0(stem, ".alignment.json")), "--strict"))
    if (status != 0) stop("Measured facet alignment failed")
  } else {
    jsonlite::write_json(list(verdict = "PASS", backend = "native_R_measured_facet_edges",
                             n_panels = length(panels), tolerance_pt = 1.5),
                        paste0(stem, ".alignment.json"), auto_unbox = TRUE, pretty = TRUE)
  }
}

save_plot <- function(plot, stem, width_mm, height_mm) {
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
  ragg::agg_tiff(paste0(stem, ".tiff"), width = width_mm, height = height_mm,
                 units = "mm", res = 600, compression = "lzw")
  print(plot)
  grDevices::dev.off()
}

x_extent <- range(pairs$background_fitness_mean)
x_limits <- x_extent + c(-1, 1) * max(diff(x_extent) * .05, .015)
log_extent <- log2(range(c(1, pairs$mfe_lower, pairs$mfe_upper, pairs$posterior_mean)))
y_limits <- 2 ^ (log_extent + c(-1, 1) * max(diff(log_extent) * .05, .025))
y_breaks <- sort(unique(c(1, scales::breaks_extended(n = 5)(y_limits))))
y_breaks <- y_breaks[y_breaks > y_limits[1] & y_breaks < y_limits[2]]
pages <- split(seq_along(features), ceiling(seq_along(features) / 13))
plots <- list()
audits <- list()
for (page in seq_along(pages)) {
  selected <- features[pages[[page]]]
  d <- pairs[pairs$interaction %in% selected, , drop = FALSE]
  d$interaction <- factor(d$interaction, levels = selected)
  plot <- ggplot(d, aes(background_fitness_mean, posterior_mean, colour = mutation_1)) +
    geom_hline(yintercept = 1, colour = "#777777", linewidth = .35, linetype = "dashed") +
    geom_errorbar(aes(ymin = mfe_lower, ymax = mfe_upper), width = 0, alpha = .35, linewidth = .25) +
    geom_point(size = 1.05, alpha = .9) +
    scale_colour_manual(values = palette, guide = "none") +
    scale_x_continuous(limits = x_limits, breaks = scales::breaks_extended(n = 4),
                       labels = scales::label_number(accuracy = .01), expand = expansion(mult = 0)) +
    scale_y_continuous(transform = "log2", limits = y_limits, breaks = y_breaks,
                       labels = scales::label_number(accuracy = .01), expand = expansion(mult = 0)) +
    facet_wrap(vars(interaction), ncol = 3, scales = "fixed", labeller = as_labeller(panel_labels)) +
    labs(x = "Fitness without focal interaction", y = "Interaction MFE (log scale)")
  built <- ggplot_build(plot)
  marks <- built$data[[3]]
  intervals <- built$data[[2]]
  stopifnot(nrow(marks) == nrow(d), nrow(intervals) == nrow(d),
            max(abs(marks$x - d$background_fitness_mean)) < 1e-12,
            max(abs(marks$y - log2(d$posterior_mean))) < 1e-12,
            max(abs(intervals$ymin - log2(d$mfe_lower))) < 1e-12,
            max(abs(intervals$ymax - log2(d$mfe_upper))) < 1e-12)
  stem <- file.path(out_dir, sprintf("interaction_mfe_backgrounds_page_%02d", page))
  require_facet_panel_alignment(plot, built$layout$layout, stem, 183, 230)
  save_plot(plot, stem, 183, 230)
  plots[[page]] <- plot
  audits[[page]] <- data.frame(page = page, interaction = selected,
                               n_backgrounds = as.integer(table(d$interaction)), all_intervals_retained = TRUE)
}
grDevices::pdf(file.path(out_dir, "interaction_mfe_all_backgrounds.pdf"), width = 183 / 25.4,
               height = 230 / 25.4, family = "Helvetica", useDingbats = FALSE, onefile = TRUE)
for (plot in plots) print(plot)
grDevices::dev.off()

# Carrier-weighted summary: average ratios within each draw, then summarize.
summary$interaction <- factor(summary$interaction, levels = rev(features))
half <- ceiling(length(features) / 2)
summary$block <- factor(ceiling(seq_len(nrow(summary)) / half), labels = c(
  sprintf("Interactions 1-%d", half), sprintf("Interactions %d-%d", half + 1, length(features))))
overview_labels <- setNames(sprintf("%02d  %s \u00d7\n%s", seq_along(features),
                                    display(summary$mutation_1), display(summary$mutation_2)), features)
overview_range <- log2(range(c(1, summary$mfe_lower, summary$mfe_upper, summary$posterior_mean)))
overview_limits <- 2 ^ (overview_range + c(-1, 1) * max(diff(overview_range) * .06, .025))
overview <- ggplot(summary, aes(posterior_mean, interaction)) +
  geom_vline(xintercept = 1, colour = "#777777", linewidth = .3, linetype = "dashed") +
  geom_segment(aes(x = mfe_lower, xend = mfe_upper, yend = interaction), linewidth = .4, colour = "#7C9DAF") +
  geom_point(size = 1.2, colour = "#0072B2") +
  scale_x_continuous(transform = "log2", limits = overview_limits,
                     breaks = sort(unique(c(1, scales::breaks_extended(n = 4)(overview_limits)))),
                     labels = scales::label_number(accuracy = .01)) +
  scale_y_discrete(labels = overview_labels) +
  facet_wrap(vars(block), ncol = 2, scales = "free_y") +
  labs(x = "Carrier-weighted interaction MFE (log scale)", y = NULL) +
  theme(axis.text.y = element_text(size = 6.5, lineheight = .95),
        panel.grid.major.y = element_blank(), panel.spacing = grid::unit(12, "pt"))
built <- ggplot_build(overview)
stopifnot(nrow(built$data[[2]]) == length(features), nrow(built$data[[3]]) == length(features),
          max(abs(built$data[[3]]$x - log2(summary$posterior_mean))) < 1e-12,
          max(abs(built$data[[2]]$x - log2(summary$mfe_lower))) < 1e-12,
          max(abs(built$data[[2]]$xend - log2(summary$mfe_upper))) < 1e-12)
stem <- file.path(out_dir, "interaction_mfe_overview")
require_facet_panel_alignment(overview, built$layout$layout, stem, 183, 245)
save_plot(overview, stem, 183, 245)
audit <- do.call(rbind, audits)
stopifnot(nrow(audit) == length(features), !anyDuplicated(audit$interaction), sum(audit$n_backgrounds) == nrow(pairs))
write.csv(audit, file.path(out_dir, "panel_audit.csv"), row.names = FALSE)
write.csv(pairs, file.path(out_dir, "background_source_data.csv"), row.names = FALSE)
write.csv(summary, file.path(out_dir, "overview_source_data.csv"), row.names = FALSE)
jsonlite::write_json(list(candidate = metadata$candidate, backend = "R/ggplot2", n_interactions = length(features),
                         n_background_estimates = nrow(pairs), n_background_pages = length(pages),
                         n_posterior_draws = metadata$n_total_draws, n_excluded_interactions = 0,
                         n_excluded_backgrounds = 0, interval = "equal_tailed_95_percent",
                         background_point_estimator = "posterior_mean_draw_paired_ratio",
                         overview_point_estimator = "posterior_mean_of_carrier_weighted_ratio_within_draw",
                         shared_background_x_limits = x_limits, shared_background_y_limits = y_limits,
                         overview_x_limits = overview_limits, posterior_sha256 = metadata$input_hashes$posterior_sha256,
                         sampler_diagnostics = metadata$sampler_diagnostics, R_version = R.version.string,
                         figure_contract = list(question = "How does each explicit interaction change predicted genetic fitness across sampled co-carrier backgrounds?",
                                                claim = "Interaction ablation ratios and posterior uncertainty are shown for every fitted interaction",
                                                archetype = "quantitative grids; paired background effects and carrier-weighted overview")),
                    file.path(out_dir, "figure_metadata.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
writeLines(c("# Explicit interaction effects under the fitted full monotone GP", "",
             "For each sampled genotype carrying both mutations, MFE = full-model genetic fitness divided by fitness after setting only the focal interaction coefficient to zero. Main effects, all other interactions, the GP mapping and posterior draw are held fixed. There is no refitting, coefficient renormalization, PIP threshold or credible-interval filter. Branch random effects are excluded.", "",
             sprintf("Background panels show all %d interaction/genotype combinations for %d fitted interactions. Points are posterior mean draw-paired ratios; vertical bars show equal-tailed 95%% credible intervals from all %s posterior draws. The x coordinate is posterior mean genetic fitness with the focal interaction disabled; x uncertainty is not displayed. Every panel shares the same axis ranges. All points denote observed double-carrier genotypes; interaction-off predictions are model ablations rather than observed genotypes. Colour follows the first named mutation. Dashed lines mark MFE = 1.", nrow(pairs), length(features), format(metadata$n_total_draws, big.mark = ",")), "",
             "The overview averages interaction fitness ratios across co-carrier backgrounds, weighted by sampled carrier counts within each posterior draw, then shows the posterior mean and equal-tailed 95% interval for that aggregate. Each interaction appears exactly once. Main and interaction names refer to the encoded features; compound indicators remain intact.", "",
             "The effect shown is an explicit interaction-coefficient ablation. The four-genotype epistasis ratio F11*F00/(F10*F01) also reflects nonlinear global epistasis and is a different estimand.", "",
             sprintf("Sampler diagnostics: %d chains, %d retained draws per chain, %d divergences. Figure rendering does not establish posterior convergence.", metadata$n_chains, metadata$draws_per_chain, metadata$sampler_diagnostics$divergences)),
           file.path(out_dir, "figure_legend.md"))
message("Saved all ", length(features), " interaction MFEs: overview and ", length(pages),
        " background pages; all ", nrow(pairs), " estimates and intervals retained")
