# Changelog

All notable changes to the ltu-ili package will be documented in this file.

## [0.0.2] - 2024-02-09

### Added
- Added customization functionality to the `ili.validation.metrics.PlotSinglePosterior` metric. 
   - Added ability to set seaborn-like `plot_kws` when calling PlotSinglePosterior
   - Added ability to overplot multiple posteriors from different models
   - Added ability to specify lower and upper limits for corner plot subplots
   - Default PlotSinglePosterior error bars are shown at central 68% and 95% percentile intervals.

### Changed
- `sbi` version restricted to be <=0.22.0

### Fixed
- N/A


## [0.0.1] - 2024-02-09

### Added
- Initial release of the project.

### Changed
- N/A

### Fixed
- N/A
