// Roda os cenários que o pytest gera no RaceView do navegador (frontend/race_view.js) e devolve o
// estado final de cada um. tests/test_frontend.py compara com o cliente Python: os dois têm de dar
// exatamente o mesmo resultado.
const RaceView = require("../../frontend/race_view.js");

const scenarios = JSON.parse(require("fs").readFileSync(0, "utf8"));
const results = scenarios.map(({ snapshot, events }) => {
  const view = new RaceView();
  if (snapshot) view.applySnapshot(snapshot);
  for (const event of events) view.applyEvent(event);
  return { epoch: view.epoch, race: view.race, cars: view.cars, classification: view.classification() };
});
process.stdout.write(JSON.stringify(results));
