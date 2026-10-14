"""Matemática do ritmo do replay: relaciona o tempo da CORRIDA com o relógio da PAREDE.

O erro clássico é "dormir a diferença entre eventos": `sleep((t[i+1] - t[i]) / speed)` em loop.
Cada sleep atrasa um pouco (o agendador do SO, o tempo gasto publicando) e os atrasos SOMAM: depois
de 1 hora o replay está segundos atrás, e trocar a velocidade no meio empilha mais erro.

Aqui o horário de cada evento é calculado a partir de uma ÂNCORA, nunca acumulado:

    tempo_de_corrida(agora) = âncora_corrida + (agora - âncora_parede) * velocidade
    evento vence quando      tempo_de_corrida(agora) >= t_evento

Se um evento atrasa 3 ms, o seguinte NÃO herda o atraso: o horário dele continua sendo a mesma
fórmula. Trocar a velocidade ou pausar apenas MOVE a âncora para o instante atual (o tempo de
corrida presente vira a nova âncora), então não há salto nem deriva. Esta classe é pura (recebe o
"agora" como argumento, não lê relógio nenhum), por isso é testada com números, sem asyncio.
"""

import math


def validate_speed(speed: float) -> float:
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError(f"velocidade deve ser um número positivo e finito, recebi {speed!r}")
    return float(speed)


class Timeline:
    def __init__(self, origin_ms: int, now: float, speed: float = 1.0) -> None:
        self._anchor_race_ms = float(origin_ms)  # tempo de corrida no instante da âncora
        self._anchor_wall_s = now  # relógio monotônico (s) no instante da âncora
        self._speed = validate_speed(speed)
        self._paused = False

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def paused(self) -> bool:
        return self._paused

    def race_ms(self, now: float) -> float:
        """Tempo de corrida "de agora". Pausado, fica congelado na âncora."""
        if self._paused:
            return self._anchor_race_ms
        return self._anchor_race_ms + (now - self._anchor_wall_s) * 1000.0 * self._speed

    def delay_s(self, event_ms: int, now: float) -> float | None:
        """Quantos segundos de parede faltam para o evento vencer (<= 0: já venceu).
        `None` se pausado: o evento não vence enquanto o tempo da corrida está parado."""
        if self._paused:
            return None
        return (event_ms - self.race_ms(now)) / (1000.0 * self._speed)

    def _reanchor(self, now: float) -> None:
        self._anchor_race_ms = self.race_ms(now)
        self._anchor_wall_s = now

    def set_speed(self, speed: float, now: float) -> None:
        speed = validate_speed(speed)
        self._reanchor(now)  # congela o passado na velocidade antiga ANTES de trocar
        self._speed = speed

    def pause(self, now: float) -> None:
        if not self._paused:
            self._reanchor(now)
            self._paused = True

    def resume(self, now: float) -> None:
        if self._paused:
            self._paused = False
            self._anchor_wall_s = now  # a âncora de corrida ficou congelada; só a da parede anda
