// Executa o script INLINE de frontend/index.html no node, com um DOM de mentira e um WebSocket
// falso, entrega uma mensagem `strategy` e devolve o que a página escreveu na tela. Pega o erro de
// digitação que só o navegador pegaria (um id que não existe, uma função que não é chamada).
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const html = fs.readFileSync(path.join(__dirname, "../../frontend/index.html"), "utf8");
const script = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]).pop();
const RaceView = require("../../frontend/race_view.js");

const elements = {};
const node = (id) => {
  const el = (elements[id] ||= {
    id, textContent: "", className: "", hidden: false, disabled: false, value: "", checked: false,
    children: [], scrollTop: 0, clientHeight: 0, scrollHeight: 0,
    addEventListener() {}, appendChild(child) { this.children.push(child); },
    replaceChildren(...kids) { this.children = kids; },
  });
  return el;
};
const intervals = [];
const sockets = [];
class FakeSocket {
  constructor(url) { this.url = url; sockets.push(this); }
  close() {}
}
const context = {
  RaceView, console, JSON, Object, Number, String, Boolean, Math, Promise, Array,
  document: { getElementById: node, createElement: () => node(`__${Math.random()}`) },
  location: { protocol: "http:", host: "localhost:8000" },
  WebSocket: FakeSocket,
  fetch: async () => ({ ok: false, json: async () => ({}) }),
  setInterval: (fn) => intervals.push(fn),
  setTimeout: () => 0,
  clearTimeout: () => {},
};
vm.createContext(context);
vm.runInContext(script, context);

const input = JSON.parse(fs.readFileSync(0, "utf8"));
const socket = sockets[0];
socket.onopen();
socket.onmessage({ data: JSON.stringify(input.snapshot) });
for (const message of input.messages) socket.onmessage({ data: JSON.stringify(message) });
intervals[0](); // o render, que a página agenda a cada 250 ms

const text = (id) => node(id).textContent;
process.stdout.write(
  JSON.stringify({
    empty: text("st-empty"),
    bodyHidden: node("st-body").hidden,
    honesty: text("st-honesty"),
    reliability: text("st-reliability"),
    reliabilityClass: node("st-reliability").className,
    lap: text("st-lap"),
    rows: node("st-rows").children.map((tr) => tr.children.map((td) => td.textContent)),
    leaderRows: node("st-rows").children.map((tr) => tr.className),
    alerts: node("st-alerts").children.map((li) => li.textContent),
    warnings: node("st-warnings").children.map((li) => li.textContent),
    unsupportedHidden: node("st-unsupported").hidden,
  }),
);
