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

## 自动更新（GitHub Pages 推荐）

仓库已配置 **GitHub Actions**，报告会**自动更新**，分享的链接始终是最新数据：

- **定时**：每天北京时间上午 10:00 自动生成报告并推送到仓库，Pages 会随之更新。
- **手动**：在 GitHub 仓库 → **Actions** → 选择「更新天气报告」→ **Run workflow**，可立即触发一次更新。

无需自己再运行脚本或手动 push，[https://ellinlin02-star.github.io/guangzhou-weather-report/](https://ellinlin02-star.github.io/guangzhou-weather-report/) 会保持更新。

---

## 更新报告后（手动时）

- **GitHub Pages**：改完代码或重新运行 `python3 guangzhou_weather_report.py` 后，把 `docs/index.html` 的变更提交并 `git push`，Pages 会自动更新。
- **Netlify 拖拽**：重新拖拽一次 `docs` 文件夹即可覆盖旧版本。

---

---

## 智能客服 · 大模型自然语言查数

报告页右下角有**智能客服**，支持用自然语言提问（如「2025年最热哪天」「近10年升温多少」「今天适合穿什么」）。

- **不配置时**：使用内置 FAQ 关键词匹配，回答较固定。
- **接入大模型后**：回答更自然，可基于报告数据做真实「查数」。

### 接入步骤（推荐用免费 Groq，或 OpenAI）

1. **部署问答接口到 Vercel**
   - 将本仓库导入 [Vercel](https://vercel.com)（或只部署含 `api/` 的目录）。
   - **免费方案（推荐）**：在项目 **Settings → Environment Variables** 中新增 `GROQ_API_KEY`。到 [Groq Console](https://console.groq.com) 注册即可免费获取 API Key，无需信用卡。
   - **付费方案**：若使用 OpenAI，则新增 `OPENAI_API_KEY`（你的 OpenAI API Key）。接口会优先使用 Groq，未配置 Groq 时才用 OpenAI。
   - 部署后得到域名，如 `https://guangzhou-weather-report.vercel.app`，则问答接口为 `https://guangzhou-weather-report.vercel.app/api/chat`。

2. **让报告页使用该接口**
   - **方式 A（推荐）**：在 GitHub 仓库 **Settings → Secrets and variables → Actions** 中新增 Secret，名称 `CHAT_API_URL`，值填 `https://你的项目.vercel.app/api/chat`。之后每次 Actions 自动更新报告时，生成的页面都会带上该地址，客服即可使用大模型。
   - **方式 B**：本地生成报告时在终端执行  
     `CHAT_API_URL=https://你的项目.vercel.app/api/chat python3 guangzhou_weather_report.py`，再把生成的 `docs/index.html` 提交并推送。

3. **仅用 Vercel 托管报告时**  
   若把整站部署到 Vercel（根目录设为 `docs`），报告与接口同域，可将 Secret `CHAT_API_URL` 设为 `/api/chat`，生成的页面会请求同站接口，无需填完整 URL。

接口约定：前端对 `/api/chat` 发送 **POST**，Body 为 `{"question": "用户问题", "reportContext": "报告摘要文本"}`，返回 `{"answer": "大模型回答"}`。报告摘要在生成时已写入页面，无需额外配置。

---

## 注意事项

- 页面通过 CDN 加载 Chart.js 和 XLSX.js，需联网访问。
- 若需国内访问更稳定，可考虑 **Gitee  Pages** 或 **Vercel**，步骤类似：指定站点根目录为 `docs` 或 `docs/index.html` 所在目录即可。
