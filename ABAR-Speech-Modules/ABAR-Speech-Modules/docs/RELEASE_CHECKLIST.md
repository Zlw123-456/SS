# Before Publishing

This package was prepared for local review. No remote repository has been created and no files have been uploaded.

- [ ] Confirm that only the intended speech BA and BR additions are present.
- [ ] Obtain supervisor/institution approval for copyright attribution and a license; then add the approved `LICENSE` file.
- [ ] Confirm software author order in `CITATION.cff`.
- [ ] Confirm the manuscript's final name and public citation. Do not mark the work accepted or invent a DOI.
- [ ] Check the module settings against the final speech experiments, not older ablation configurations.
- [ ] Confirm whether the final paper link may be public; add it only after approval.
- [ ] Run `python -m unittest discover -s tests -v` in a clean environment with PyTorch installed.
- [ ] Run both examples and inspect the GitHub Actions result after creating the repository.
- [ ] Recheck the archive for model weights, datasets, private paths, secrets, and third-party backbone source.
- [ ] Create the repository only after publication approval. Add a verified repository URL to the README/CFF afterward.

## Suggested GitHub description

Standalone band allocation and band recalibration modules for speech separation; no TIGER backbone code or pretrained weights.

## Suggested topics

`speech-separation`, `band-allocation`, `feature-recalibration`, `pytorch`, `audio`

## Suggested repository name

`ABAR-Speech-Modules`

## 中文提醒

当前包只供本地审核。开源协议留给老师确认，不要在创建仓库时随手勾选 MIT 或其他协议。不要把原始 TIGER 项目、训练权重、日志、数据集或服务器配置一起拖进去。

上传时使用本包目录内的文件，使 `README.md` 位于 GitHub 仓库根目录，而不是只上传 ZIP。保留 `.github`、`.gitignore` 等点开头的文件。若老师决定更换作者顺序或论文标题，先修改 `CITATION.cff` 和 README，再发布。
