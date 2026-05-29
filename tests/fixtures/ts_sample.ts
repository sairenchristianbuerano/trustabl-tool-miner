import { tool } from "@anthropic-ai/claude-agent-sdk";
import { z } from "zod";

// Claude Agent SDK factory form: tool(name, description, schema, handler).
export const fetchData = tool(
  "fetch_data",
  "Fetches data from a URL",
  { url: z.string() },
  async (args) => {
    const r = await fetch(args.url);
    return await r.json();
  },
);

// Missing description + shell exec.
export const runCmd = tool(
  "run_cmd",
  "",
  { cmd: z.string() },
  async (args) => {
    execSync(args.cmd);
    return "ok";
  },
);

// OpenAI Agents JS object-config form.
const sendEmail = tool({
  name: "send_email",
  description: "Send an email",
  parameters: z.object({ to: z.string() }),
  execute: async (input) => {
    return await axios.post("https://api.example.com/send", input);
  },
});
