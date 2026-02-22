"""Email alerts via SMTP."""

from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

logger = logging.getLogger(__name__)


class EmailAlert:
    """Send prediction alerts via SMTP email."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        sender_email: str,
        sender_password: str,
        recipient_email: str,
        use_tls: bool = True,
    ) -> None:
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_email = recipient_email
        self.use_tls = use_tls

    def send_predictions(
        self,
        predictions: list[dict[str, Any]],
        model_name: str = "ensemble",
        run_id: str = "",
    ) -> bool:
        """Send formatted prediction email."""
        if not predictions:
            return True

        subject = f"MLB GBSV — {len(predictions)} Picks ({model_name})"
        html = self._build_html(predictions, model_name, run_id)
        return self._send(subject, html)

    def send_daily_summary(
        self,
        stats: dict[str, Any],
        top_picks: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Send daily performance summary."""
        roi = stats.get("roi", 0)
        subject = f"MLB GBSV Daily — ROI: {roi:.1%}"
        html = self._build_summary_html(stats, top_picks)
        return self._send(subject, html)

    def send_alert(self, subject: str, message: str) -> bool:
        """Send a simple alert email."""
        html = f"<html><body><p>{message}</p></body></html>"
        return self._send(subject, html)

    # -- private --------------------------------------------------------------

    def _build_html(
        self,
        predictions: list[dict[str, Any]],
        model_name: str,
        run_id: str,
    ) -> str:
        rows = ""
        for p in predictions:
            prob = p.get("probability", 0.5)
            side = "HOME" if p.get("prediction", 0) == 1 else "AWAY"
            bg = "#d4edda" if prob >= 0.6 else "#fff3cd" if prob >= 0.55 else "#f8f9fa"
            rows += f"""
            <tr style="background:{bg}">
                <td>{p.get('away_team', '?')} @ {p.get('home_team', '?')}</td>
                <td><b>{side}</b></td>
                <td>{p.get('spread', 0):+.1f}</td>
                <td>{prob:.1%}</td>
                <td>{p.get('edge', 0):.1%}</td>
                <td>${p.get('recommended_bet', 0):,.0f}</td>
            </tr>"""

        return f"""
        <html><body>
        <h2>MLB GBSV Predictions — {model_name}</h2>
        <p>Run: <code>{run_id}</code></p>
        <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse">
            <tr style="background:#343a40;color:#fff">
                <th>Game</th><th>Pick</th><th>Spread</th>
                <th>Confidence</th><th>Edge</th><th>Bet Size</th>
            </tr>
            {rows}
        </table>
        <p style="color:#888;font-size:12px">MLB GBSV v1.0</p>
        </body></html>
        """

    def _build_summary_html(
        self,
        stats: dict[str, Any],
        top_picks: list[dict[str, Any]] | None,
    ) -> str:
        roi_color = "#28a745" if stats.get("roi", 0) >= 0 else "#dc3545"
        picks_html = ""
        if top_picks:
            for p in top_picks[:5]:
                picks_html += (
                    f"<li>{p.get('home_team', '?')} vs {p.get('away_team', '?')} "
                    f"— {p.get('probability', 0):.0%} conf</li>"
                )

        return f"""
        <html><body>
        <h2>MLB GBSV Daily Summary</h2>
        <table cellpadding="8">
            <tr><td>Total Bets</td><td><b>{stats.get('total_bets', 0)}</b></td></tr>
            <tr><td>Win Rate</td><td><b>{stats.get('win_rate', 0):.1%}</b></td></tr>
            <tr><td>ROI</td>
                <td style="color:{roi_color}"><b>{stats.get('roi', 0):.1%}</b></td></tr>
            <tr><td>Net Profit</td>
                <td><b>${stats.get('net_profit', 0):,.2f}</b></td></tr>
            <tr><td>Balance</td>
                <td><b>${stats.get('current_balance', 0):,.2f}</b></td></tr>
            <tr><td>Sharpe</td><td>{stats.get('sharpe', 0):.3f}</td></tr>
        </table>
        {"<h3>Top Picks</h3><ul>" + picks_html + "</ul>" if picks_html else ""}
        </body></html>
        """

    def _send(self, subject: str, html_body: str) -> bool:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.sender_email
        msg["To"] = self.recipient_email
        msg.attach(MIMEText(html_body, "html"))

        try:
            if self.use_tls:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port)
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port)

            server.login(self.sender_email, self.sender_password)
            server.sendmail(self.sender_email, self.recipient_email, msg.as_string())
            server.quit()
            logger.info("Email sent: %s", subject)
            return True
        except Exception as exc:
            logger.error("Email send failed: %s", exc)
            return False
