/* Regras do protocolo do WebSocket de corrida (as mesmas de scripts/watch_race.py, a classe RaceView).
 *
 * É um arquivo à parte da página para poder ser testado no node: tests/test_frontend.py roda a MESMA
 * sequência de eventos neste código e no cliente Python e exige o mesmo estado final.
 *
 *  - O 1º frame é o `snapshot`. Depois chegam eventos, que PODEM repetir o que o snapshot já refletia.
 *  - `epoch` maior = replay novo (descarta tudo); menor = replay antigo (ignora).
 *  - Por piloto: uma posição só vale se t > pos_t; uma volta, se t > lap_t.
 *  - Status da corrida só avança em (volta, estado): started < running < finished.
 *  - Classificação: mais voltas primeiro; empatando, quem cruzou antes; sem volta, por último.
 *  - `strategy` (Fase 3): vale o de MAIOR `lap` do epoch atual, e só se `v` é uma versão conhecida
 *    (senão fica em `strategyUnsupported`: dizer ao usuário, não adivinhar). Sempre mostre
 *    `baseline_win` ao lado de `win`: no backtest o Monte Carlo não superou "o líder atual vence".
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.RaceView = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const STATE_RANK = { started: 0, running: 1, finished: 2 };
  const STRATEGY_VERSION = 1; // a versão do formato do evento `strategy` que esta página entende

  class RaceView {
    constructor() {
      this.epoch = 0;
      this.race = {};
      this.cars = {};
      this.strategy = null;
      this.strategyUnsupported = null;
    }

    applySnapshot(message) {
      if (this.strategy && this.strategy.epoch !== message.epoch) this.strategy = null; // era de outro replay
      this.epoch = message.epoch;
      this.race = Object.assign({}, message.race);
      this.cars = {};
      for (const [code, car] of Object.entries(message.cars)) this.cars[code] = Object.assign({}, car);
    }

    applyEvent(message) {
      if (message.epoch < this.epoch) return; // evento de um replay antigo
      if (message.epoch > this.epoch) {
        this.epoch = message.epoch; // replay novo: recomeça
        this.race = {};
        this.cars = {};
        this.strategy = null;
        this.strategyUnsupported = null;
      }
      switch (message.type) {
        case "strategy": {
          if (message.v !== STRATEGY_VERSION) this.strategyUnsupported = message.v;
          else if (this.strategy === null || message.lap > this.strategy.lap) this.strategy = message;
          break;
        }
        case "position": {
          const car = (this.cars[message.driver] ||= {});
          if ((car.pos_t ?? -1) < message.t) {
            car.x = message.x;
            car.y = message.y;
            car.pos_t = message.t;
          }
          break;
        }
        case "lap_completed": {
          const car = (this.cars[message.driver] ||= {});
          if ((car.lap_t ?? -1) < message.t) {
            car.lap = message.lap;
            car.lap_t = message.t;
            for (const name of ["lap_time_ms", "compound", "position"]) {
              if (message[name] === null || message[name] === undefined) delete car[name];
              else car[name] = message[name]; // volta sem esse dado não herda o da anterior
            }
          }
          break;
        }
        case "race_status": {
          const lap = this.race.lap ?? 0;
          const rank = STATE_RANK[this.race.state] ?? -1;
          const incoming = STATE_RANK[message.state];
          if (message.lap > lap || (message.lap === lap && incoming > rank)) {
            this.race = { state: message.state, lap: message.lap };
            if (message.total_laps !== null && message.total_laps !== undefined)
              this.race.total_laps = message.total_laps;
          }
          break;
        }
      }
    }

    get finished() {
      return this.race.state === "finished";
    }

    classification() {
      const codes = Object.keys(this.cars);
      const withLaps = codes.filter((c) => "lap" in this.cars[c]);
      const without = codes.filter((c) => !("lap" in this.cars[c])).sort();
      withLaps.sort((a, b) => {
        const ca = this.cars[a];
        const cb = this.cars[b];
        return cb.lap - ca.lap || (ca.lap_t ?? 0) - (cb.lap_t ?? 0) || (a < b ? -1 : a > b ? 1 : 0);
      });
      return withLaps.concat(without);
    }
  }

  // ---- o que a página mostra do evento `strategy`, sem DOM (testável no node)

  const pct = (x) => (x * 100).toFixed(1) + "%";
  const seconds = (x) => (x >= 0 ? "+" : "") + x.toFixed(1) + " s";

  // A ressalva está SEMPRE na tela, qualquer que seja a volta: o backtest (docs/backtest.md) não
  // achou o Monte Carlo melhor que "o líder atual vence" em nenhuma fase da corrida.
  const HONESTY =
    "O Monte Carlo NÃO superou “o líder atual vence” no backtest (21 corridas de 2024, docs/backtest.md): " +
    "compare as duas colunas e não trate a probabilidade dele como melhor que a posição na pista.";
  const RELIABILITY_TEXT = {
    indicative:
      "Até 80% da corrida o Monte Carlo não se distinguiu de “o líder atual vence” no backtest.",
    experimental:
      "EXPERIMENTAL (a partir de 80% da corrida): no backtest o Monte Carlo foi PIOR que “o líder atual vence” nesta fase.",
  };
  const ALERT_NOTE =
    "Alertas medidos só nas paradas que aconteceram: a precisão é otimista (docs/undercut-ao-vivo.md). O líder nunca é o rival.";

  function strategyView(strategy, unsupportedVersion) {
    if (unsupportedVersion !== null && unsupportedVersion !== undefined && !strategy)
      return { empty: `versão ${unsupportedVersion} do formato de estratégia não suportada por esta página` };
    if (!strategy) return { empty: "sem resultado de estratégia ainda (o cálculo começa na volta 5)" };
    const f = strategy.forecast;
    return {
      reliability: strategy.reliability,
      honesty: HONESTY,
      reliabilityText: RELIABILITY_TEXT[strategy.reliability] ?? "",
      lapLine: `volta ${strategy.lap}/${strategy.total_laps}, faltam ${strategy.laps_left}`,
      leader: strategy.baseline.leader,
      rows: f.drivers.map((d) => ({
        driver: d.driver,
        mc: pct(d.win),
        baseline: pct(d.baseline_win),
        podium: pct(d.podium),
        leader: d.driver === strategy.baseline.leader,
      })),
      alerts: strategy.alerts.map(
        (a) =>
          `${a.attacker} sobre ${a.rival ?? "?"}: p=${pct(a.probability ?? 0)}, margem ${a.margin_s == null ? "?" : seconds(a.margin_s)} (K=${a.horizon_k ?? "?"})`,
      ),
      alertNote: ALERT_NOTE,
      warnings: strategy.warnings.map((w) => w.text),
      inputs: `carros que reportaram: ${strategy.inputs.cars_reported}/${strategy.inputs.cars_known}; voltas atrasadas: ${strategy.inputs.late_laps}`,
      unsupported: unsupportedVersion !== null && unsupportedVersion !== undefined,
    };
  }

  RaceView.strategyView = strategyView;
  return RaceView;
});
