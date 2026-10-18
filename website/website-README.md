# Public website source

source.zip contains only the static public website source (HTML, CSS, images, public result CSVs and the static build scripts). GitHub Actions unpacks it and publishes only the allowlisted static output. Registration and submission are separate pages on the Tencent Cloud server; no API client or backend source is included.

Maintain the local frontend/ directory, update the public data CSVs there, rebuild source.zip and replace this archive to publish. Existing task1/, task2/ and Releases are unchanged.
