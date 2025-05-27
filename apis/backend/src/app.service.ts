// import { spawn } from 'child_process';
import { ConfigService } from '@nestjs/config';
import { resolve } from 'path';
import { PythonShell } from 'python-shell';
import { firstValueFrom } from 'rxjs';
import { HttpService } from '@nestjs/axios';
import { Injectable } from '@nestjs/common';
import { AnalysisService } from './analysis.service';

@Injectable()
export class AppService {

  ANALYSIS_SERVICES: string;

  constructor(private config: ConfigService,
    private httpService: HttpService,
    private readonly analysisService: AnalysisService,
  ) {
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
          const parsed: any = fullResponse;
          if (parsed.error) {
            reject(parsed);
          } else {
            resolve(parsed);
            // this.analysisLLM(parsed)
            // .then((response: string) => {
            //   resolve(response);
            // })
            //   .catch((e: any) => {
            //     console.error('LLM Analysis Error:', e);
            //     reject('Failed to analyze with LLM');
            //     return;
            //   })
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

  async analyzeAcademicYearEnhanced(year: number): Promise<any> {
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
          const parsed = JSON.parse(fullResponse);
          if (parsed.error) {
            reject(parsed);
          } else {
            this.analysisService.analysisLLM(parsed)
              .then((response: string) => {
                const result = {
                  ...parsed,
                  year,
                  analysis: response
                }
                resolve(result);
              })
              .catch((e: any) => {
                console.error('LLM Analysis Error:', e);
                const result = {
                  ...parsed,
                  year,
                  analysis: 'Failed to analyze with LLM'
                }
                reject(result);
                return;
              })
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
          const parsed: any = fullResponse;
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

  private async analysisLLM(query: string) {
    try {
      const prompt = this.buildPrompt(query);

      // Add debug logs right after creating the prompt
      console.log("=== LLM DEBUG INFO ===");
      console.log("Prompt length (chars):", prompt.length);
      console.log("Estimated prompt tokens:", Math.ceil(prompt.length / 4));
      console.log("First 200 chars:", prompt.substring(0, 200) + (prompt.length > 200 ? "..." : ""));

      const response: any = await firstValueFrom(this.httpService.post(
        `${process.env.LLM_SERVER_URL}:${process.env.LLM_SERVER_PORT}${process.env.LLM_API_ENDPOINT_COMPLETION}`,
        {
          prompt: prompt,
          n_predict: 500,  // Increased from 256
          // temperature: 0.2,
          temperature: 0.1,
          top_k: 30,
          top_p: 0.85,
          stop: ["</s>", "[INST]", "\n\n"], // Mistral-specific stops
          // stop: ["\n\n", "6.", "RESPONSE:"], // Stop after 5 points
          repeat_penalty: 1.5, // Reduce repetition,
          repeat_last_n: 0,  // Prevent repetition
          mirostat: 2,       // Better response quality
          mirostat_tau: 5.0,
          mirostat_eta: 0.1
        },
        // { timeout: 3000 } // 10 second timeout
      ));

      // Add these debug logs:
      console.log("Prompt length (chars):", prompt.length);
      console.log("Estimated prompt tokens:", Math.ceil(prompt.length / 4));

      if (!response) {
        throw new Error('No response received from the LLM server.');
      }

      const { data } = response;

      // Enhanced token logging
      console.log("=== LLM RESPONSE INFO ===");
      console.log("Actual tokens used:", data?.tokens_predicted);
      console.log("Response length:", data?.content?.length);
      console.log("First 100 chars of response:", data?.content?.substring(0, 100));

      console.log("Actual tokens used:", data?.tokens_predicted);

      return this.cleanResponse(data?.content); // <-- Apply cleaning here
      // return this.formatResponse(data?.content.trim() || "I couldn't generate a response.");
    }
    catch (error) {
      console.error('LLM Query Error:', error);
      return this.withRetry(() => this.analysisLLM(query));
    }
  }

  private buildPrompt(results: string): string {
    const prompt = `
    Analyze this student performance report and summarize key insights:
    ${JSON.stringify(results)}
    
    Focus on:
    - Underperforming students
    - Subject-wise weaknesses
    - Recommended actions
    `;

    return prompt;
  }
  private cleanResponse(raw: string): string {
    if (!raw) return "I couldn't generate a response.";

    // Remove duplicate numbered points
    const lines = raw.split('\n');
    const uniqueLines: string[] = [];
    const seenPoints = new Set();

    for (const line of lines) {
      const pointMatch = line.match(/^\d+\./);
      if (pointMatch) {
        const pointText = line.replace(/^\d+\.\s*/, '').trim().toLowerCase();
        if (seenPoints.has(pointText)) continue;
        seenPoints.add(pointText);
      }
      uniqueLines.push(line);
    }

    // Enforce maximum 5 points
    const filteredLines = uniqueLines.filter((line, index) => {
      const pointNum = parseInt(line.match(/^(\d+)\./)?.[1] || "0");
      return pointNum <= 5 || !line.match(/^\d+\./);
    });

    return filteredLines.join('\n').trim();
  }
  private async withRetry<T>(fn: () => Promise<T>, retries = 2): Promise<T> {
    try {
      return await fn();
    } catch (error) {
      if (retries > 0) {
        await new Promise(res => setTimeout(res, 1000));
        return this.withRetry(fn, retries - 1);
      }
      throw error;
    }
  }

  async checkLLMHealth() {
    try {
      // Debug env vars first
      console.log('Current environment:', {
        NODE_ENV: process.env.NODE_ENV,
        LLM_SERVER_URL: process.env.LLM_SERVER_URL,
        LLM_SERVER_PORT: process.env.LLM_SERVER_PORT,
        LLM_API_ENDPOINT_HEALTH: process.env.LLM_API_ENDPOINT_HEALTH
      });

      const baseUrl = process.env.LLM_SERVER_URL || 'http://localhost';
      const port = process.env.LLM_SERVER_PORT || '8080';
      const endpoint = process.env.LLM_API_ENDPOINT_HEALTH || '/health';

      // Remove duplicate slashes
      const healthCheckUrl = `${baseUrl}:${port}${endpoint}`.replace(/([^:]\/)\/+/g, '$1');

      console.log('Final health check URL:', healthCheckUrl);

      const response = await firstValueFrom(
        this.httpService.get(healthCheckUrl, { timeout: 5000 })
      );

      return response?.data?.status === "ok";
    } catch (error) {
      console.error('Health check failed:', error.message);
      return false;
    }
  }




}
