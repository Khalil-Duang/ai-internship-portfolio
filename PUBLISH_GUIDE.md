# GitHub 发布说明

## 推荐方式

优先做成一个干净的“作品集主页仓库”，再为每个项目保留独立目录。若后续愿意继续迭代，也可以拆成三个独立仓库，并在主页 README 中放项目链接。

不建议把整个 Codex 工作目录、课程压缩包、原始数据集一次性上传。招聘方更在意项目能否快速看懂、运行和验证。

## 上传前检查

1. 简历 PDF 放在本地求职资料中，不放进公开仓库，避免手机号被搜索引擎收录。
2. 不提交 `data/`、`results/`、个人数据表、`*.xls`、`*.csv`、SQL Server 数据文件和日志。
3. 将机器名、数据库名、姓名拼音和学号等个人标识替换成通用占位符。
4. 只保留与项目有关的代码、README 和脱敏样例；课程作业说明不等于你的个人结论，应补充自己的复盘。
5. 第一次提交前运行 `git status`，确认没有数据文件、密钥、密码、令牌或 `.env`。
6. 若仓库公开，建议使用邮箱作为联系方式，手机号只放在投递简历中。

## 发布命令

```bash
git init
git add .
git commit -m "Add data and AI internship portfolio"
git branch -M main
git remote add origin https://github.com/<你的用户名>/data-ai-portfolio.git
git push -u origin main
```

若希望展示个人主页，可新建一个与 GitHub 用户名同名的仓库，把 `profile-README.md` 的内容放到该仓库的 `README.md`。

## 建议的仓库简介

Data Science and AI internship portfolio: NLP, data preprocessing, and SQL Server ETL projects.

