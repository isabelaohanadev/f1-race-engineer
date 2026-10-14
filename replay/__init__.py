"""Replay de uma corrida gravada no banco como se fosse ao vivo (Fase 2).

    banco ──► RaceSource ──► ReplayEngine ──► StreamPublisher ──► Redis Stream `race:{y}:{r}`
              (keyset)       (Timeline+Clock)  (XADD MAXLEN ~)

* events.py     contrato dos eventos no stream (position, lap_completed, race_status).
* source.py     lê o banco em ordem de tempo, em páginas (keyset), sem carregar a corrida inteira.
* timeline.py   matemática pura do ritmo: tempo de corrida x relógio, velocidade, pausa.
* clock.py      relógio monotônico injetável (testes rodam sem sleeps reais).
* engine.py     o loop de replay: espera cada evento chegar a sua hora e publica.
* publisher.py  XADD em lote com tamanho máximo do stream.
* lock.py       lock no Redis: um único replay por corrida, mesmo com vários processos.
* manager.py    registro dos replays em andamento (start/pause/resume/speed).
"""
