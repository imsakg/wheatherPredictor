# History reconstruction

The three original repository commits (`663825c`, `e639ea2`, `6489267`) are preserved unchanged.

The subsequent 57 local development commits are a **later reconstruction**, intentionally assigned author and committer dates between July 2021 and June 2022 at the repository owner's request. Their dates are not contemporaneous proof of implementation, publication, releases, API availability or dependency versions. Modern tooling in this tree should not be interpreted as having existed on those historical dates.

Each added commit contains a substantive implementation, regression test, integration example, documentation or CI change; none is an empty contribution-calendar filler. New author dates were supplied through `git commit --date`, then committer dates aligned by rebasing only the new range. Original commits were not rewritten. Nothing was pushed as part of the reconstruction.

The maintained implementation is the `weather_predictor` package. Original datasets are retained; original root-level experimental scripts are guarded against import-time side effects rather than represented as a validated neural forecasting system.
