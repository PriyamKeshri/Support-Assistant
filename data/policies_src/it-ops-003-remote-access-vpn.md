# Remote Access and VPN Policy
Policy ID: IT-OPS-003 | Version 4.0 | Owner: Network Engineering | Effective: 1 April 2026

## 1. Scope
This policy applies to every connection to Kestrel Analytics internal systems from outside an office network.

## 2. The KestrelConnect VPN
KestrelConnect is the company VPN client. It is based on WireGuard and is installed automatically on every managed laptop. You need it to reach internal systems such as the data warehouse, the Git servers and admin consoles. SaaS tools behind single sign-on, such as email, Teams and Jira, work without the VPN. On public or hotel Wi-Fi, KestrelConnect must be switched on for all traffic.

## 3. Split Tunneling
Split tunneling is off by default, so all traffic goes through the VPN while you are connected. Microsoft Teams and Zoom media traffic is excluded automatically to protect call quality. Any other exception needs an Access Request approved by the Network Engineering team.

## 4. Troubleshooting VPN Errors
- VPN-809 (handshake timeout): your network is blocking UDP port 51820. Switch to a different network or a mobile hotspot; hotel and guest networks often block this port.
- VPN-412 (certificate expired): your device certificate has expired, usually because the laptop has not checked in for more than 30 days. Connect to the internet, open Company Portal, choose Sync, then reconnect.
- VPN-127 (MFA rejected): the MFA prompt was denied or timed out. Prompts expire after 60 seconds, so retry and approve the prompt within that time.
- VPN-503 (gateway unavailable): check status.kestrel.example for a known outage before raising a ticket.
If an error continues, raise a ticket in the Network/VPN category and include the error code.

## 5. Working From Abroad
Tell IT at least 10 business days before you work from another country, using the Travel Notice form in the service portal. Working remotely from abroad is limited to 30 days per calendar year unless HR approves more in writing. For travel to countries on Information Security's high-risk list, you must take a clean loaner laptop instead of your primary laptop.

## 6. Home Network Requirements
Your home Wi-Fi must use WPA2-AES or WPA3 with a non-default password, and the router firmware must be kept up to date. Do not let other people use your work laptop, and do not connect it by USB to shared or public computers.
