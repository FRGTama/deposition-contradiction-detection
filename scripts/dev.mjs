import { spawn } from "node:child_process";

const commands = [
  {
    name: "api",
    command: "python",
    args: ["-m", "uvicorn", "backend.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
  },
  {
    name: "web",
    command: "vite",
    args: ["--host", "0.0.0.0", "--port", "5173"]
  }
];

const children = commands.map(({ name, command, args }) => {
  const child = spawn(command, args, {
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
    shell: true,
    stdio: ["inherit", "pipe", "pipe"]
  });

  child.stdout.on("data", (chunk) => process.stdout.write(`[${name}] ${chunk}`));
  child.stderr.on("data", (chunk) => process.stderr.write(`[${name}] ${chunk}`));
  child.on("exit", (code, signal) => {
    if (shuttingDown) {
      return;
    }
    console.error(`[${name}] exited with ${signal ?? code}`);
    shutdown(code ?? 1);
  });

  return child;
});

let shuttingDown = false;

function shutdown(code = 0) {
  shuttingDown = true;
  for (const child of children) {
    if (!child.killed) {
      child.kill();
    }
  }
  process.exit(code);
}

process.on("SIGINT", () => shutdown(0));
process.on("SIGTERM", () => shutdown(0));
