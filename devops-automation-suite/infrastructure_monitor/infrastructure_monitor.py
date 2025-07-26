"""
DevOps Automation Suite - Infrastructure Monitor
Advanced Infrastructure Health Monitoring and Auto-scaling
"""

import psutil
from datetime import datetime
from typing import List, Dict, Any
from loguru import logger
from core.config import settings

class InfrastructureMonitor:
    """
    Infrastructure Health Monitoring Class
    """
    
    def __init__(self):
        self.alert_cpu_threshold = settings.alert_cpu_threshold
        self.alert_memory_threshold = settings.alert_memory_threshold
        self.alert_disk_threshold = settings.alert_disk_threshold
        logger.info("Initialized Infrastructure Monitor")

    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get system resource usage metrics
        """
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
        logger.info(f"System metrics collected: {metrics}")
        return metrics

    def check_thresholds(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Check if metrics exceed alert thresholds
        """
        alerts = []
        if metrics["cpu_percent"] > self.alert_cpu_threshold:
            alerts.append("CPU usage alert")
        if metrics["memory_percent"] > self.alert_memory_threshold:
            alerts.append("Memory usage alert")
        if metrics["disk_percent"] > self.alert_disk_threshold:
            alerts.append("Disk usage alert")
        logger.info(f"Alerts generated: {alerts}")
        return alerts

    def monitor(self):
        """
        Main monitoring loop
        """
        while True:
            metrics = self.get_system_metrics()
            alerts = self.check_thresholds(metrics)
            # TODO: Implement alert mechanism (e.g. notify via Slack/Email)
            # Sleep for the monitoring interval (converted to seconds)
            psutil.time.sleep(settings.monitoring_interval)


def start_infrastructure_monitor():
    """
    Start infrastructure monitoring
    """
    monitor = InfrastructureMonitor()
    monitor.monitor()
