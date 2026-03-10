# BCN-XMU-course2026 作业提交仓库

## 提交规则
1. 每个小组只能往自己的专属文件夹上传代码（例：小组01 → group01/）；
2. 禁止修改/删除其他小组文件夹的内容；
3. 代码命名用英文，禁止中文（例：main_analysis.py，不要用「主分析.py」）；
4. 截止时间：2026-04-02 23:59。

## 上传方法
### 方式1：网页端上传（推荐新手）
1. 进入自己的小组文件夹（例：group01/）；
2. 点击「Add file」→「Upload files」；
3. 拖拽代码文件到页面，填写提交说明（例：「group01 提交第1次作业」）；
4. 点击「Commit changes」完成上传。

### 方式2：命令行上传（适合有Git基础的同学）
```bash
# 1. 克隆仓库到本地
git clone https://github.com/QuantLet/BCN-XMU-course2026.git
cd BCN-XMU-course2026

# 2. 进入自己的小组文件夹，放入代码
cd group01
# 把代码复制到这个文件夹里

# 3. 提交并上传
git add .
git commit -m "group01: submit homework 1"
git push origin main
