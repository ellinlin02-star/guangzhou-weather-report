/**
 * 广州天气报告 · 大模型问答接口
 * 支持免费方案与付费方案二选一（优先使用免费）：
 *   - Groq（免费）：在 Vercel 环境变量中设置 GROQ_API_KEY，到 https://console.groq.com 注册即可获取。
 *   - OpenAI（付费）：设置 OPENAI_API_KEY。
 * 前端将 CHAT_API_URL 设为 /api/chat（同域）或 https://你的项目.vercel.app/api/chat（跨域）。
 */

const SYSTEM_PROMPT = `你是「广州天气数据报告」的智能助手。请严格根据下面提供的报告摘要数据，用简短、自然的中文回答用户问题。
回答要求：
- 只基于给出的报告数据回答，不要编造数字或日期；若数据中没有相关信息，如实说明「报告里没有这部分信息」。
- 语气友好、简洁，适合普通用户阅读。
- 若用户问的是「查数」类问题（如某年最热哪天、某指标数值），请直接给出数据中的答案。`;

// 优先 Groq（免费），其次 OpenAI
function getProvider() {
  if (process.env.GROQ_API_KEY) {
    return {
      type: "groq",
      apiKey: process.env.GROQ_API_KEY,
      url: "https://api.groq.com/openai/v1/chat/completions",
      model: "llama-3.1-8b-instant",
    };
  }
  if (process.env.OPENAI_API_KEY) {
    return {
      type: "openai",
      apiKey: process.env.OPENAI_API_KEY,
      url: "https://api.openai.com/v1/chat/completions",
      model: "gpt-4o-mini",
    };
  }
  return null;
}

module.exports = async function (req, res) {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }

  if (req.method !== "POST") {
    return res.status(405).json({ error: "请使用 POST 请求" });
  }

  const provider = getProvider();
  if (!provider) {
    return res.status(500).json({
      error: "请至少配置一种大模型：在 Vercel 环境变量中设置 GROQ_API_KEY（免费，到 console.groq.com 注册）或 OPENAI_API_KEY。",
    });
  }

  let body;
  try {
    body = typeof req.body === "string" ? JSON.parse(req.body) : req.body || {};
  } catch (e) {
    return res.status(400).json({ error: "请求体不是合法 JSON" });
  }

  const { question, reportContext } = body;
  if (!question || typeof question !== "string") {
    return res.status(400).json({ error: "缺少 question 或格式错误" });
  }

  const reportText =
    typeof reportContext === "string" && reportContext.trim()
      ? reportContext.trim()
      : "（暂无报告摘要）";

  const userMessage = `【报告摘要】\n${reportText}\n\n【用户问题】\n${question.trim()}`;

  try {
    const response = await fetch(provider.url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${provider.apiKey}`,
      },
      body: JSON.stringify({
        model: provider.model,
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: userMessage },
        ],
        max_tokens: 600,
        temperature: 0.3,
      }),
    });

    if (!response.ok) {
      const errText = await response.text();
      return res.status(response.status).json({
        error: `大模型服务异常: ${response.status}`,
        detail: errText.slice(0, 200),
      });
    }

    const data = await response.json();
    const content =
      data.choices?.[0]?.message?.content?.trim() ||
      "未能生成回答，请重试。";

    return res.status(200).json({ answer: content });
  } catch (e) {
    return res.status(500).json({
      error: "请求大模型时出错",
      detail: e.message || String(e),
    });
  }
}
