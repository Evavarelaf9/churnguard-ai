"""Mapeamento ORM (SQLAlchemy 2.0) das tabelas definidas em database/migrations.

O schema é a fonte da verdade (migrations SQL); estes modelos apenas o espelham
para leitura e escrita tipadas.
"""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func

JSONType = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class Plan(Base):
    __tablename__ = "plans"
    plano_id: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    plano: Mapped[str] = mapped_column(String(40), unique=True)
    preco_base: Mapped[Decimal] = mapped_column(Numeric(10, 2))
    funcionalidades_disponiveis: Mapped[int] = mapped_column(SmallInteger)
    usuarios_incluidos: Mapped[int] = mapped_column(SmallInteger)


class PriceAdjustment(Base):
    __tablename__ = "price_adjustments"
    ano: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    percentual: Mapped[Decimal] = mapped_column(Numeric(5, 4))
    descricao: Mapped[str | None] = mapped_column(String(200))


class Customer(Base):
    __tablename__ = "customers"
    customer_id: Mapped[str] = mapped_column(String(12), primary_key=True)
    nome: Mapped[str] = mapped_column(String(120))
    idade: Mapped[int | None] = mapped_column(SmallInteger)
    regiao: Mapped[str | None] = mapped_column(String(20))
    tipo_cliente: Mapped[str] = mapped_column(String(30))
    plano_id: Mapped[int] = mapped_column(ForeignKey("plans.plano_id"))
    ciclo_contrato: Mapped[str] = mapped_column(String(10))
    forma_pagamento: Mapped[str] = mapped_column(String(30))
    renovacao_automatica: Mapped[bool] = mapped_column(Boolean)
    canal_aquisicao: Mapped[str] = mapped_column(String(30))
    data_inicio: Mapped[date] = mapped_column(Date)
    data_cancelamento: Mapped[date | None] = mapped_column(Date)
    motivo_cancelamento: Mapped[str | None] = mapped_column(String(60))
    status: Mapped[str] = mapped_column(String(12), default="Ativo")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    plan: Mapped[Plan] = relationship(lazy="joined")


class UsageMonthly(Base):
    __tablename__ = "usage_monthly"
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"), primary_key=True)
    mes_referencia: Mapped[date] = mapped_column(Date, primary_key=True)
    acessos: Mapped[int] = mapped_column(Integer)
    sessoes: Mapped[int] = mapped_column(Integer)
    minutos_uso: Mapped[int] = mapped_column(Integer)
    funcionalidades_usadas: Mapped[int] = mapped_column(SmallInteger)
    dias_ativos: Mapped[int] = mapped_column(SmallInteger)
    ultimo_acesso: Mapped[date | None] = mapped_column(Date)


class Payment(Base):
    __tablename__ = "payments"
    payment_id: Mapped[str] = mapped_column(String(14), primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"))
    mes_referencia: Mapped[date] = mapped_column(Date)
    valor_cobrado: Mapped[Decimal] = mapped_column(Numeric(10, 2))
    data_vencimento: Mapped[date] = mapped_column(Date)
    data_pagamento: Mapped[date | None] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(20))


class Ticket(Base):
    __tablename__ = "tickets"
    ticket_id: Mapped[str] = mapped_column(String(12), primary_key=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"))
    data_abertura: Mapped[date] = mapped_column(Date)
    categoria: Mapped[str] = mapped_column(String(20))
    tempo_resolucao_horas: Mapped[Decimal | None] = mapped_column(Numeric(8, 1))
    avaliacao: Mapped[int | None] = mapped_column(SmallInteger)
    data_resolucao: Mapped[date | None] = mapped_column(Date)


class Survey(Base):
    __tablename__ = "surveys"
    survey_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"))
    data_resposta: Mapped[date] = mapped_column(Date)
    nota_satisfacao: Mapped[int] = mapped_column(SmallInteger)


class ModelVersion(Base):
    __tablename__ = "model_versions"
    version: Mapped[str] = mapped_column(String(20), primary_key=True)
    model_name: Mapped[str] = mapped_column(String(80))
    trained_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    threshold: Mapped[Decimal] = mapped_column(Numeric(4, 3))
    risk_thresholds: Mapped[dict] = mapped_column(JSONType)
    metrics: Mapped[dict] = mapped_column(JSONType)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONType)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"))
    data_referencia: Mapped[date] = mapped_column(Date)
    model_version: Mapped[str] = mapped_column(ForeignKey("model_versions.version"))
    probabilidade: Mapped[Decimal] = mapped_column(Numeric(6, 5))
    nivel_risco: Mapped[str] = mapped_column(String(10))
    threshold: Mapped[Decimal] = mapped_column(Numeric(4, 3))
    principal_fator: Mapped[str | None] = mapped_column(Text)
    categoria_principal: Mapped[str | None] = mapped_column(String(20))
    fatores: Mapped[dict | None] = mapped_column(JSONType)
    recomendacao: Mapped[dict | None] = mapped_column(JSONType)
    entrada: Mapped[dict | None] = mapped_column(JSONType)
    origem: Mapped[str] = mapped_column(String(12))
    churn_real: Mapped[int | None] = mapped_column(SmallInteger)
    resultado_em: Mapped[date | None] = mapped_column(Date)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RetentionAction(Base):
    __tablename__ = "retention_actions"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"))
    prediction_id: Mapped[int | None] = mapped_column(ForeignKey("predictions.id"))
    tipo_acao: Mapped[str] = mapped_column(String(30))
    status: Mapped[str] = mapped_column(String(15), default="Planejada")
    responsavel: Mapped[str | None] = mapped_column(String(80))
    observacoes: Mapped[str | None] = mapped_column(Text)
    resultado: Mapped[str | None] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class AccessLog(Base):
    __tablename__ = "access_log"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    usuario: Mapped[str] = mapped_column(String(80))
    acao: Mapped[str] = mapped_column(String(40))
    customer_id: Mapped[str | None] = mapped_column(String(12))
    detalhe: Mapped[dict | None] = mapped_column(JSONType)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
