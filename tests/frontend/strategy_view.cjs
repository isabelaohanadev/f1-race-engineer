// Roda `RaceView.strategyView` (o que a página mostra do evento `strategy`) para os casos que o
// pytest gera e devolve o resultado. tests/test_frontend.py confere o texto e os números.
const RaceView = require("../../frontend/race_view.js");

const cases = JSON.parse(require("fs").readFileSync(0, "utf8"));
process.stdout.write(
  JSON.stringify(cases.map(({ strategy, unsupported }) => RaceView.strategyView(strategy, unsupported))),
);
