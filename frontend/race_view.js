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
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.RaceView = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const STATE_RANK = { started: 0, running: 1, finished: 2 };

  class RaceView {
    constructor() {
      this.epoch = 0;
      this.race = {};
      this.cars = {};
    }

    applySnapshot(message) {
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
      }
      switch (message.type) {
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

  return RaceView;
});
