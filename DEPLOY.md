# 广州天气报告 · 线上部署说明

报告为**单页静态 HTML**（数据已内嵌），可直接部署到任意静态托管。每次运行 `guangzhou_weather_report.py` 会同时生成根目录的 `guangzhou_weather_report.html` 和 **`docs/index.html`**，后者用于部署。

---

## 方式一：GitHub Pages（免费）

1. **若尚未使用 Git**，在项目目录执行：
   ```bash
   git init
   ```

2. **在 GitHub 新建仓库**（如 `guangzhou-weather-report`），不要勾选 “Add a README”（若已勾选可忽略本步）。

3. **关联远程并推送**（将 `你的用户名`、`guangzhou-weather-report` 换成你的信息）：
   ```bash
   git add .
   git commit -m "广州天气报告"
   git branch -M main
   git remote add origin https://github.com/你的用户名/guangzhou-weather-report.git
   git push -u origin main
   ```

4. **开启 GitHub Pages**  
   - 仓库 → **Settings** → **Pages**  
   - **Source** 选 **Deploy from a branch**  
   - **Branch** 选 `main`，**Folder** 选 **/docs**  
   - 保存后等待 1～2 分钟

5. **访问地址**：`https://你的用户名.github.io/guangzhou-weather-report/`

---

## 方式二：Netlify（免费，拖拽即可）

1. 打开 [https://app.netlify.com/drop](https://app.netlify.com/drop)
2. 将本地的 **`docs`** 文件夹（内含 `index.html`）拖入页面
3. 部署完成后会得到一个随机 URL（如 `https://xxx.netlify.app`），可自定义子域名

**或** 若项目已在 GitHub：在 Netlify 中 **Add new site → Import from Git**，选该仓库，**Publish directory** 填 `docs`，部署即可。

---

## 更新报告后

- **GitHub Pages**：改完代码或重新运行 `python3 guangzhou_weather_report.py` 后，把 `docs/index.html` 的变更提交并 `git push`，Pages 会自动更新。
- **Netlify 拖拽**：重新拖拽一次 `docs` 文件夹即可覆盖旧版本。

---

## 注意事项

- 页面通过 CDN 加载 Chart.js 和 XLSX.js，需联网访问。
- 若需国内访问更稳定，可考虑 **Gitee  Pages** 或 **Vercel**，步骤类似：指定站点根目录为 `docs` 或 `docs/index.html` 所在目录即可。
