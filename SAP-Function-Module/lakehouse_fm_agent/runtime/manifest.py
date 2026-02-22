
import json, time

class Manifest:
    def __init__(self, plan_id:str):
        self.plan_id = plan_id
        self.items = []
    def ensure_table(self, catalog, schema, name, reason, depends_on):
        self.items.append({
            'plan_id': self.plan_id,
            'object_type': 'TABLE',
            'catalog_name': catalog,
            'schema_name': schema,
            'object_name': name,
            'fm_source': 'plan',
            'planned_action': 'CREATE',
            'reason': reason,
            'depends_on': depends_on,
            'status': 'PLANNED',
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'created_by': 'fmtool'
        })
    def to_json(self):
        return json.dumps(self.items, indent=2)
