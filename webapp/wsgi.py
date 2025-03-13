from main import create_app

app = create_app()

if __name__ == "__main__":
    from waitress import serve
    import os
    
    port = int(os.getenv("PORT", 5000))
    serve(app, host='0.0.0.0', port=port, threads=4) 