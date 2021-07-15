from os import getenv
from Project2API.Configurations.APIConfiguration import CreateApp
from Project2API.Configurations.RouterConfiguration import ConfigRoutes
from Project2API.Extensions.ResponseExtension import Ok

app = CreateApp(getenv('FLASK_ENV') or 'Default')
ConfigRoutes(app)

@app.route("/")
def StatusApi():
    return Ok({'Status': 'Running'})

if __name__ == '__main__':
    host = app.config['HOST']
    port = app.config['APP_PORT']
    debug = app.config['DEBUG']

    app.run(host=host, debug=debug, port=port, use_reloader=debug)
