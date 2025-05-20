import { Injectable } from '@nestjs/common';
// import { spawn } from 'child_process';
import { ConfigService } from '@nestjs/config';
import { resolve } from 'path';
import { PythonShell } from 'python-shell';

@Injectable()
export class AppService {

  ANALYSIS_SERVICES: string;

  constructor(private config: ConfigService) {
    // Initialize any necessary properties or services here
    const analysisServicePath = this.config.get('ANALYSIS_SERVICE_PATH');

    this.ANALYSIS_SERVICES = resolve(process.cwd(), analysisServicePath);
  }

  getHello(): string {
    return 'Hello World!';
  }

  // via python shell
  async analyzeAcademicYear(year: number): Promise<any> {
    // const options = {
    //   pythonPath: `${this.ANALYSIS_SERVICES}/venv/bin/python`, // Direct path
    //   args: [JSON.stringify(year.toString())],
    //   env: {
    //     ...process.env,
    //   }
    // };

    const options: any = {
      pythonPath: `${this.ANALYSIS_SERVICES}/venv/bin/python`,
      scriptPath: this.ANALYSIS_SERVICES,
      args: [year.toString()], // Don't stringify twice
      mode: 'text'
    };

    return new Promise((resolve, reject) => {
      const shell = new PythonShell(`analyzer.py`, options);

      let fullResponse = '';

      shell.on('message', (message: string) => {
        fullResponse += message;
        console.log('Raw Python message:', fullResponse); // Debug log
      });

      shell.end((err: any) => {
        if (err) {
          reject({ error: err.message });
          return;
        }

        try {
          // const parsed = JSON.parse(fullResponse);
          const parsed : any = fullResponse;
          if (parsed.error) {
            reject(parsed);
          } else {
            resolve(parsed);
          }
        } catch (e) {
          reject({
            error: 'Invalid JSON response',
            raw: fullResponse
          });
        }
      });
    });
  }


  // via python shell
  async analyzeAcademicYearKmeans(year: number): Promise<any> {
    // const options = {
    //   pythonPath: `${this.ANALYSIS_SERVICES}/venv/bin/python`, // Direct path
    //   args: [JSON.stringify(year.toString())],
    //   env: {
    //     ...process.env,
    //   }
    // };

    const options: any = {
      pythonPath: `${this.ANALYSIS_SERVICES}/venv/bin/python`,
      scriptPath: this.ANALYSIS_SERVICES,
      args: [year.toString()], // Don't stringify twice
      mode: 'text'
    };

    return new Promise((resolve, reject) => {
      const shell = new PythonShell(`analyzerKMean.py`, options);

      let fullResponse = '';

      shell.on('message', (message: string) => {
        fullResponse += message;
        console.log('Raw Python message:', fullResponse); // Debug log
      });

      shell.end((err: any) => {
        if (err) {
          reject({ error: err.message });
          return;
        }

        try {
          // const parsed = JSON.parse(fullResponse);
          const parsed : any = fullResponse;
          if (parsed.error) {
            reject(parsed);
          } else {
            resolve(parsed);
          }
        } catch (e) {
          reject({
            error: 'Invalid JSON response',
            raw: fullResponse
          });
        }
      });
    });
  }
  // Via child process
  // async analyzeAcademicYear(year: number): Promise<any> {
  //   const python = spawn('python3', ['ai_service/analyzer.py', year.toString()]);

  //   return new Promise((resolve) => {
  //     let result = '';
  //     python.stdout.on('data', (data) => result += data.toString());
  //     console.log('# result=>#', result);
  //     python.on('close', () => resolve(JSON.parse(result)));
  //   });
  // }
}
