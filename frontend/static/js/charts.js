// Smart Lender - Chart.js Visualizations

function initDashboardCharts(statsData, modelsData) {
    // 1. Status Distribution Doughnut Chart
    const statusCtx = document.getElementById('statusChart');
    if (statusCtx && statsData) {
        new Chart(statusCtx, {
            type: 'doughnut',
            data: {
                labels: ['Approved', 'Rejected'],
                datasets: [{
                    data: [statsData.approved_count || 0, statsData.rejected_count || 0],
                    backgroundColor: ['#10b981', '#f43f5e'],
                    borderColor: ['#ffffff', '#ffffff'],
                    borderWidth: 2,
                    hoverOffset: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            font: { family: 'Inter', weight: '600' },
                            padding: 18
                        }
                    }
                },
                cutout: '72%'
            }
        });
    }

    // 2. Model Performance Comparison Grouped Bar Chart
    const modelsCtx = document.getElementById('modelsChart');
    if (modelsCtx && modelsData && modelsData.algorithms) {
        const algos = Object.keys(modelsData.algorithms);
        const accuracyData = algos.map(a => (modelsData.algorithms[a].accuracy * 100).toFixed(1));
        const precisionData = algos.map(a => (modelsData.algorithms[a].precision * 100).toFixed(1));
        const recallData = algos.map(a => (modelsData.algorithms[a].recall * 100).toFixed(1));
        const f1Data = algos.map(a => (modelsData.algorithms[a].f1_score * 100).toFixed(1));

        new Chart(modelsCtx, {
            type: 'bar',
            data: {
                labels: algos,
                datasets: [
                    {
                        label: 'Accuracy (%)',
                        data: accuracyData,
                        backgroundColor: '#0d9488',
                        borderRadius: 4
                    },
                    {
                        label: 'Precision (%)',
                        data: precisionData,
                        backgroundColor: '#3b82f6',
                        borderRadius: 4
                    },
                    {
                        label: 'Recall (%)',
                        data: recallData,
                        backgroundColor: '#8b5cf6',
                        borderRadius: 4
                    },
                    {
                        label: 'F1-Score (%)',
                        data: f1Data,
                        backgroundColor: '#10b981',
                        borderRadius: 4
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            callback: value => value + '%'
                        },
                        grid: {
                            color: '#f1f5f9'
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            font: { family: 'Inter', weight: '600' },
                            boxWidth: 14,
                            padding: 14
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.dataset.label + ': ' + context.parsed.y + '%';
                            }
                        }
                    }
                }
            }
        });
    }

    // 3. Property Area Distribution Chart
    const areaCtx = document.getElementById('areaChart');
    if (areaCtx && statsData && statsData.area_distribution) {
        const areas = Object.keys(statsData.area_distribution);
        const counts = Object.values(statsData.area_distribution);

        new Chart(areaCtx, {
            type: 'bar',
            data: {
                labels: areas,
                datasets: [{
                    label: 'Applications',
                    data: counts,
                    backgroundColor: ['#0d9488', '#38bdf8', '#a855f7'],
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: { stepSize: 1 },
                        grid: { color: '#f1f5f9' }
                    },
                    x: {
                        grid: { display: false }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }
}
