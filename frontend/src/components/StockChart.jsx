import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { fetchStockData } from '../api/stockApi';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const StockChart = ({ ticker, period }) => {
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        const data = await fetchStockData(ticker, period);
        
        const chartConfig = {
          labels: data.dates,
          datasets: [
            {
              label: `${ticker} Close Price`,
              data: data.prices.close,
              fill: false,
              backgroundColor: 'rgba(75, 192, 192, 0.2)',
              borderColor: 'rgba(75, 192, 192, 1)',
              tension: 0.1,
            },
          ],
        };
        
        setChartData(chartConfig);
        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    };

    loadData();
  }, [ticker, period]);

  if (loading) return <div className="text-center py-10">Loading stock data...</div>;
  if (error) return <div className="text-center text-red-500 py-10">Error: {error}</div>;
  if (!chartData) return null;

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: `${ticker} Stock Price (${period})`,
      },
    },
    scales: {
      x: {
        ticks: {
          maxTicksLimit: 15,
        },
      },
    },
  };

  return (
    <div className="bg-white p-4 rounded-lg shadow">
      <Line data={chartData} options={options} />
    </div>
  );
};

export default StockChart;
