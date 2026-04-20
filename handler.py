def handler(event):
    if not event:
        return {'ok': False, 'error': 'empty event'}
    return {'ok': True, 'event_id': event.get('id')}
