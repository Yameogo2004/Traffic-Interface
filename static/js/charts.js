// Fonction pour afficher un graphique Plotly
function showChart(chartId, data) {
    var layout = {
        title: data.title,
        xaxis: { title: data.xLabel },
        yaxis: { title: data.yLabel },
        margin: { t: 50, l: 50, r: 30, b: 50 }
    };

    Plotly.newPlot(chartId, data.traces, layout);
}
