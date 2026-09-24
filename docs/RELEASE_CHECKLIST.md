# Before Publishing

This update was prepared locally for the existing [SS repository](https://github.com/Zlw123-456/SS). It has not been committed or uploaded by this update process. Review differences against the current repository before merging; preserve unrelated edits and any approved license.

- [ ] Confirm that only the intended speech BA and BR additions are present.
- [ ] Obtain supervisor/institution approval for copyright attribution and a license; then add the approved `LICENSE` file.
- [ ] Confirm software author order in `CITATION.cff`.
- [ ] Confirm the manuscript's final name and public citation. Do not mark the work accepted or invent a DOI.
- [ ] Check the module settings against the final speech experiments, not older ablation configurations.
- [ ] Keep BR's code, equations, diagrams, and retrained results consistent: SiLU, `2 * sigmoid`, zero biases, and small random final-layer weights.
- [ ] Confirm whether the final paper link may be public; add it only after approval.
- [ ] Run `python -m unittest discover -s tests -v` in a clean environment with PyTorch installed.
- [ ] Run both examples and inspect the GitHub Actions result after uploading the update.
- [ ] Recheck the archive for model weights, datasets, private paths, secrets, and third-party backbone source.
- [ ] Merge only approved changes into the existing repository; retain its author information and unrelated work.

## Suggested GitHub description

Standalone band allocation and band recalibration modules for speech separation; no TIGER backbone code or pretrained weights.

## Suggested topics

`speech-separation`, `band-allocation`, `feature-recalibration`, `pytorch`, `audio`

## 中文提醒

当前更新仅在本地完成，尚未提交至现有 GitHub 仓库。开源协议仍留给老师确认，不要自动新增或覆盖协议。不要把原始 TIGER 项目、训练权重、日志、数据集或服务器配置一起拖进去。

上传时使用本包目录内的文件，使 `README.md` 位于 GitHub 仓库根目录，而不是只上传 ZIP。保留 `.github`、`.gitignore` 等点开头的文件。若仓库中的作者信息、论文标题或其他内容已更新，请保留这些改动，只合并本次相关变化。代码和说明仍使用 BR 名称，不另设开发版本名称。
