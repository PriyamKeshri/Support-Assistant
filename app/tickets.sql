CREATE SEQUENCE IF NOT EXISTS ticket_number_seq START 1001;

CREATE TABLE IF NOT EXISTS tickets (
    id          bigserial PRIMARY KEY,
    number      text NOT NULL UNIQUE DEFAULT ('INC-' || nextval('ticket_number_seq')),
    requester   text NOT NULL,
    title       text NOT NULL,
    description text NOT NULL,
    category    text NOT NULL CHECK (category IN ('Account and Password', 'Access Request', 'Hardware', 'Software',
                                                  'Network/VPN', 'Email and Collaboration', 'Security Incident')),
    priority    text NOT NULL CHECK (priority IN ('P1', 'P2', 'P3', 'P4')),
    status      text NOT NULL DEFAULT 'Open'
                CHECK (status IN ('Open', 'In Progress', 'Waiting on User', 'Resolved', 'Closed')),
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS tickets_requester_idx ON tickets (requester, status);

-- Every change and comment, with who made it and who approved it: the audit trail.
CREATE TABLE IF NOT EXISTS ticket_events (
    id          bigserial PRIMARY KEY,
    ticket_id   bigint NOT NULL REFERENCES tickets (id) ON DELETE CASCADE,
    action      text NOT NULL CHECK (action IN ('created', 'updated', 'commented')),
    changes     jsonb NOT NULL DEFAULT '{}',
    comment     text,
    actor       text NOT NULL,
    approved_by text,
    created_at  timestamptz NOT NULL DEFAULT now()
);
