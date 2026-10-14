"""Erros de domínio do replay. A API (etapa seguinte) os traduz em códigos HTTP."""


class ReplayError(Exception):
    """Base de todos os erros do replay."""


class RaceNotFound(ReplayError):
    """A corrida não foi ingerida no banco."""


class RaceHasNoData(ReplayError):
    """A corrida existe, mas não tem nenhuma posição nem volta com horário para reproduzir."""


class ReplayAlreadyRunning(ReplayError):
    """Já existe um replay ativo para esta corrida (neste processo ou em outro)."""


class ReplayNotRunning(ReplayError):
    """Não há replay ativo para esta corrida (nunca iniciou, ou já terminou)."""


class InvalidSpeed(ReplayError):
    """Velocidade fora das permitidas (1x, 4x, 16x)."""
