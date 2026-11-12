# 上传到自己的 GitHub 仓库

本包在 `zhanglei-web/peekaboo-pipeline` 中放在 `portable/`。维护现有仓库时应在仓库根目录提交；不要在 `portable/` 内再初始化嵌套 Git 仓库。下面“新建空远端仓库”只适用于单独导出的源码包。上传范围是源码和文档。上层 ROS 工作区含模型、数据、构建文件等，不应把上层目录整体加入 Git。

## 新建空远端仓库

在 GitHub 创建一个空仓库，例如 `egomemory-pipeline`，先不要初始化 README，以便推送本地记录。然后在本仓库目录执行：

```bash
python scripts/check_repository.py
pytest -q
git status --short
git add .
git commit -m "Package end-to-end egomemory annotation pipeline"
git branch -M main
git remote add origin git@github.com:YOUR_USERNAME/egomemory-pipeline.git
git push -u origin main
```

首次单独发布前按自己的身份设置 `git config user.name "Your Name"` 和 `git config user.email "you@example.com"`（只作用于本仓库）。

若本地已经有初始提交，跳过 commit；若已有 origin，先用 `git remote -v` 确认，再更新它。替换自己的用户名和仓库名，认证使用本机已有的 SSH key 或 GitHub 登录，不把 token 放进源码或文档。

## 已有内容的远端仓库

先克隆现有远端到新目录，再把本包的源码和文档复制进去，保留该远端的 `.git`，检查差异后提交。不要为合并方便使用 force push 覆盖已有历史。

## 压缩包方式

交付 ZIP 包不含 `.git`、运行数据、模型、缓存或密钥。解压后可执行 `git init -b main`，再按上面的步骤上传。GitHub 网页上传也可用于源码文件，但保留目录结构和点文件更适合用 Git。

运行数据、权重与生成 JSON 使用独立数据存储；项目 README 的历史统计不依赖它们随仓库公开。仓库保留已知第三方声明，未替所有组件指定统一新许可证。
