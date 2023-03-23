library(tidyverse)
library(broom)

oils <- read_excel("HSI_PLS.xlsx")
cols <- colnames(oils)
spectra <- cols[4:length(cols)]  # As characters

# Add replicate column
oils <- oils %>%
  mutate(replicate = rep(1:3, nrow(oils)/3))

# Convert to long form
oils_long <- oils %>%
  pivot_longer(cols = spectra, names_to = "wavelength", values_to = "intensity") %>%
  mutate(wavelength = as.numeric(wavelength))
print(head(oils_long))

# Compute principal components
pca <- oils %>%
  select(all_of(spectra)) %>%
  prcomp(scale = TRUE)

# Plot "pure" oils (EVOO and 6 adulterants)
pure_oils <- oils_long %>%
  filter(str_starts(`Sample ID/Wavelength`, "Expt") |
           str_starts(`Sample ID/Wavelength`, "Pure"))

# Trace plots for pure oils
p <- ggplot(pure_oils, aes(x = wavelength, y = intensity, color = `Sample ID/Wavelength`,
                           group = interaction(`Sample ID/Wavelength`, replicate))) +
  geom_line()
print(p)

# Plot first two PCs
p <- pca %>%
  augment(oils) %>%
  filter(`% Adulteration` < 100) %>%
  mutate(adulterated = `% Adulteration` > 0) %>%
  ggplot(aes(.fittedPC1, .fittedPC2, color = `% Adulteration`, shape = adulterated)) +
  geom_point(size = 2)
print(p)

# Loadings plot

# define arrow style for plotting
arrow_style <- arrow(
  angle = 20, ends = "first", type = "closed", length = grid::unit(8, "pt")
)

# Percentage of variance explained
p <- pca %>%
  tidy(matrix = "eigenvalues") %>%
  head(n = 9) %>%
  ggplot(aes(PC, percent)) +
  geom_col(fill = "#56B4E9", alpha = 0.8) +
  scale_x_continuous(breaks = 1:9) +
  scale_y_continuous(
    labels = scales::percent_format(),
    expand = expansion(mult = c(0, 0.01))
  )
print(p)

# plot rotation matrix
summarized_pca <- pca %>%
  tidy(matrix = "rotation") %>%
  mutate(column = as.numeric(column),
         centile = cut(column, breaks = seq(900, 1800, by = 100))) %>%
  group_by(centile, PC) %>%
  summarize(mvalue = mean(value))

summarized_pca %>%
  pivot_wider(names_from = "PC", names_prefix = "PC", values_from = "mvalue") %>%
  ggplot(aes(PC1, PC2)) +
  geom_segment(xend = 0, yend = 0, arrow = arrow_style) +
  geom_text(
    aes(label = centile),
    hjust = 1, nudge_x = 0.03,
    color = "#904C2F"
  ) +
  xlim(-0.01, 0.11)

