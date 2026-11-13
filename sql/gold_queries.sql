-- Distribuição de classes da Camada Gold
SELECT koi_disposition, records, percentage
FROM gold.koi_disposition_summary
ORDER BY records DESC;

-- Estatísticas descritivas das variáveis físicas
SELECT
    koi_disposition,
    COUNT(*) AS total,
    AVG(koi_period) AS avg_orbital_period_days,
    AVG(koi_prad) AS avg_planet_radius_earth,
    AVG(koi_teq) AS avg_equilibrium_temperature_k
FROM gold.koi_analytics
GROUP BY koi_disposition
ORDER BY total DESC;

-- Dataset analítico utilizado pelo treinamento
SELECT * FROM gold.koi_analytics LIMIT 20;
