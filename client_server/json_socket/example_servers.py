"""
This entire file is simply a set of examples. The most basic is to
simply create a custom server by inheriting tserver.ThreadedServer
as shown below in MyServer.
"""

import logging

from client_server.server.json_socket import jsocket_base

from client_server.json_socket import tserver

logger = logging.getLogger("jsocket.example_servers")


class MyServer(tserver.ThreadedServer):
    """	This is a basic example of a custom ThreadedServer.	"""

    def __init__(self):
        super(MyServer, self).__init__()
        self.timeout = 2.0
        logger.warning("MyServer class in customServer is for example purposes only.")

    def _process_message(self, obj):
        """ virtual method """
        if obj != '':
            if obj['message'] == "new connection":
                logger.info("new connection.")


class MyFactoryThread(tserver.ServerFactoryThread):
    """	This is an example factory thread, which the server factory will
        instantiate for each new connection.
    """

    def __init__(self):
        super(MyFactoryThread, self).__init__()
        self.timeout = 2.0

    def _process_message(self, obj):
        """ virtual method - Implementer must define protocol """
        if obj != '':
            if obj['message'] == "new connection":
                logger.info("new connection.")
            else:
                logger.info(obj)


if __name__ == "__main__":
    import time

    server = tserver.ServerFactory(MyFactoryThread, address='127.0.0.1', port=5490)
    server.timeout = 2.0
    server.start()

    time.sleep(1)
    cPids = []
    for i in range(10):
        client = jsocket_base.JsonClient(address='127.0.0.1', port=5490)
        cPids.append(client)
        client.connect()
        client.send_obj({"message": "new connection"})
        client.send_obj({"message": i})

    time.sleep(5)

    for c in cPids:
        c.close()
    server.stop()
    server.join()
