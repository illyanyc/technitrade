from os import walk
from flask import Flask


def ConfigRoutes(app: Flask):
    for (dirpath, dirname, filenames) in walk("./Project2API/Controller"):
        for filename in filenames:
            controller = filename.replace(".py", "")
            command = f"from Project2API.Controller.{controller} import {controller}"
            exec(command)

            apiRoute = controller.replace("Controller", "").lower()
            command = f"{controller}.register(app, route_base='/{apiRoute}')"
            exec(command)
        break
